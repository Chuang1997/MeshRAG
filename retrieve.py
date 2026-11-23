"""Deterministic, corpus-only hash graph construction and candidate reranking."""
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import threading
import time

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


@dataclass(frozen=True)
class HashConfig:
    dimensions: int = 96
    tables: int = 8
    rows_per_band: int = 2
    bands_per_table: int = 6
    seed: int = 42
    neighbors: int = 20
    max_bucket: int = 256
    merge_overlap: float = 0.6
    max_community: int = 256
    initial_k: int = 20
    community_k: int = 5
    community_members: int = 100


STOP = set(ENGLISH_STOP_WORDS) | {'also', 'would', 'could', 'known', 'named'}
WORDS = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)
NAMES = re.compile(r"\b[A-Z][\w'’-]*(?:\s+(?:[A-Z][\w'’-]*|of|the|and|de|van)){0,4}\b")


def extract_entities(text):
    """Cheap regex features: content words, adjacent word pairs, and proper names.

    Word features make matching insensitive to query capitalization. These are
    lexical entities, not the output of a trained named entity recognizer.
    """
    words = [w.casefold().replace('’', "'") for w in WORDS.findall(text)]
    keep = lambda w: w not in STOP and (len(w) > 2 or w.isdigit())
    entities = {w for w in words if keep(w)}
    entities.update(a + ' ' + b for a, b in zip(words, words[1:]) if keep(a) and keep(b))
    for name in NAMES.findall(text):
        phrase = ' '.join(w.casefold().replace('’', "'") for w in WORDS.findall(name))
        if ' ' in phrase:
            entities.add(phrase)
    return entities


class MinHasher:
    prime = np.uint64(4294967291)

    def __init__(self, config):
        self.config = config
        rng = np.random.default_rng(config.seed)
        self.a = rng.integers(1, int(self.prime), config.dimensions, dtype=np.uint64)
        self.b = rng.integers(0, int(self.prime), config.dimensions, dtype=np.uint64)
        self.permutations = [rng.permutation(config.dimensions) for _ in range(config.tables)]

    def signature(self, entities):
        if not entities:
            return np.full(self.config.dimensions, int(self.prime), dtype=np.uint32)
        values = np.array([int.from_bytes(hashlib.sha256(e.encode()).digest()[:4], 'little')
                           % int(self.prime) for e in sorted(entities)], dtype=np.uint64)
        return ((values[:, None] * self.a + self.b) % self.prime).min(axis=0).astype(np.uint32)

    def keys(self, signature):
        rows = self.config.rows_per_band
        for table, permutation in enumerate(self.permutations):
            for band, start in enumerate(range(0, self.config.bands_per_table * rows, rows)):
                yield (table, band, signature[permutation[start:start + rows]].tobytes())


def order_scores(scores, candidates, limit):
    candidates = np.asarray(sorted(set(candidates)), dtype=np.int64)
    if not len(candidates):
        return []
    return candidates[np.lexsort((candidates, -scores[candidates]))[:limit]].tolist()


def merge_communities(raw, overlap=0.6, max_size=256):
    """Merge qualifying overlapping buckets to a fixed point; retain overlap.

    The inverted member index avoids comparing every pair of communities.
    """
    unique = sorted({frozenset(m) for m in raw if 2 <= len(m) <= max_size},
                    key=lambda m: (-len(m), tuple(sorted(m))))
    groups, membership = {}, defaultdict(set)
    next_id = 0
    for source in unique:
        current = set(source)
        while True:
            counts = Counter(g for member in current for g in membership[member])
            eligible = [g for g, count in counts.items()
                        if count / min(len(current), len(groups[g])) > overlap
                        and len(current | groups[g]) <= max_size]
            if not eligible:
                break
            g = min(eligible, key=lambda x: (-counts[x] / min(len(current), len(groups[x])), x))
            previous = groups.pop(g)
            for member in previous:
                membership[member].remove(g)
            current.update(previous)
        key = next_id
        next_id += 1
        groups[key] = current
        for member in current:
            membership[member].add(key)
    return sorted([sorted(m) for m in groups.values()], key=lambda m: (-len(m), m))


class HashGraph:
    def __init__(self, corpus, config=HashConfig()):
        if config.dimensions % config.rows_per_band:
            raise ValueError('Signature dimensions must be divisible by rows_per_band')
        if config.bands_per_table * config.rows_per_band > config.dimensions:
            raise ValueError('Bands exceed signature dimensions')
        self.corpus, self.config = corpus, config
        self.hasher = MinHasher(config)

    def build(self):
        started = time.monotonic()
        print('Building corpus entity signatures...', flush=True)
        entities = [extract_entities(p['text']) for p in self.corpus]
        self.signatures = np.stack([self.hasher.signature(e) for e in entities])
        buckets = defaultdict(list)
        for i, signature in enumerate(self.signatures):
            if entities[i]:
                for key in self.hasher.keys(signature):
                    buckets[key].append(i)
        # Dense, uninformative collision buckets are bounded before graph building.
        active = [members for members in buckets.values() if 2 <= len(members) <= self.config.max_bucket]
        print(f'Building chunk edges from {len(active)} collision buckets...', flush=True)
        candidate_sets = [set() for _ in self.corpus]
        for members in active:
            for i in members:
                candidate_sets[i].update(members)
        edges = {}
        for i, candidates in enumerate(candidate_sets):
            candidates.discard(i)
            if not candidates:
                continue
            scores = (self.signatures == self.signatures[i]).mean(axis=1)
            for j in order_scores(scores, candidates, self.config.neighbors):
                common = sorted(entities[i] & entities[j])
                if common:
                    edges[(min(i, j), max(i, j))] = {
                        'source': min(i, j), 'target': max(i, j),
                        'hash_similarity': float(scores[j]), 'common_entities': common}
        self.edges = list(edges.values())
        self.neighbors = [[] for _ in self.corpus]
        for a, b in edges:
            self.neighbors[a].append(b)
            self.neighbors[b].append(a)
        print('Merging overlapping collision communities...', flush=True)
        self.communities = merge_communities(active, self.config.merge_overlap, self.config.max_community)
        # A categorical majority per coordinate is compatible with MinHash equality.
        prototypes = []
        for members in self.communities:
            signatures = self.signatures[members]
            prototypes.append([int(np.unique(col, return_counts=True)[0][np.argmax(
                np.unique(col, return_counts=True)[1])]) for col in signatures.T])
        self.prototypes = np.asarray(prototypes, dtype=np.uint32).reshape(-1, self.config.dimensions)
        self.stats = {'chunks': len(self.corpus), 'edges': len(edges),
                      'communities': len(self.communities), 'raw_communities': len(active),
                      'chunks_in_communities': len({i for m in self.communities for i in m}),
                      'build_seconds': round(time.monotonic() - started, 3)}
        self._memberships()
        return self

    def _memberships(self):
        self.membership = [[] for _ in self.corpus]
        for c, members in enumerate(self.communities):
            for i in members:
                self.membership[i].append(c)

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path / 'hashes.npz', signatures=self.signatures, prototypes=self.prototypes)
        (path / 'graph.json').write_text(json.dumps({'config': asdict(self.config),
            'stats': self.stats, 'edges': self.edges, 'neighbors': self.neighbors,
            'communities': self.communities}, ensure_ascii=False))

    def load(self, path):
        path = Path(path)
        metadata = json.loads((path / 'graph.json').read_text())
        if metadata['config'] != asdict(self.config):
            raise ValueError('Cached hash configuration differs')
        arrays = np.load(path / 'hashes.npz', allow_pickle=False)
        self.signatures, self.prototypes = arrays['signatures'], arrays['prototypes']
        for key in ('stats', 'edges', 'neighbors', 'communities'):
            setattr(self, key, metadata[key])
        self._memberships()
        return self

    def candidates(self, query):
        entities = extract_entities(query)
        signature = self.hasher.signature(entities)
        scores = (self.signatures == signature).mean(axis=1) if entities else np.zeros(len(self.corpus))
        initial = order_scores(scores, range(len(self.corpus)), self.config.initial_k) if entities else []
        sources = defaultdict(set)
        for i in initial:
            sources[i].add('hash_nearest')
            for j in order_scores(scores, self.neighbors[i], self.config.neighbors):
                sources[j].add('chunk_neighbor')
        community_scores = (self.prototypes == signature).mean(axis=1) if entities else np.zeros(len(self.communities))
        # A community can be reached through its prototype or through a hash-nearest anchor.
        for rank, i in enumerate(initial):
            for c in self.membership[i]:
                community_scores[c] += 1 / (rank + 1)
        selected = order_scores(community_scores, np.flatnonzero(community_scores > 0), self.config.community_k)
        for c in selected:
            for i in order_scores(scores, self.communities[c], self.config.community_members):
                sources[i].add('community')
        return sorted(sources), {'query': query, 'query_entities': sorted(entities),
            'initial_ids': [self.corpus[i]['id'] for i in initial],
            'communities': [{'id': c, 'score': float(community_scores[c]),
                             'members': len(self.communities[c])} for c in selected],
            'candidate_count': len(sources)}, sources


class GraphRetriever:
    def __init__(self, corpus, state='state', encoder='sentence-transformers/all-MiniLM-L6-v2',
                 config=HashConfig()):
        from sentence_transformers import SentenceTransformer
        self.corpus, self.state = corpus, Path(state)
        self.state.mkdir(parents=True, exist_ok=True)
        fingerprint = hashlib.sha256(json.dumps({'corpus': corpus, 'config': asdict(config),
            'encoder': encoder, 'implementation': 1}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        metadata_path = self.state / 'manifest.json'
        if metadata_path.exists() and json.loads(metadata_path.read_text())['fingerprint'] != fingerprint:
            raise ValueError('Index inputs changed; choose an empty state directory')
        self.graph = HashGraph(corpus, config)
        if (self.state / 'graph.json').exists():
            self.graph.load(self.state)
        else:
            self.graph.build().save(self.state)
        print('Graph: ' + json.dumps(self.graph.stats), flush=True)
        self.encoder = SentenceTransformer(encoder, device='cpu')
        embeddings_path = self.state / 'embeddings.npy'
        if embeddings_path.exists():
            self.embeddings = np.load(embeddings_path, allow_pickle=False)
        else:
            self.embeddings = self.encoder.encode([p['text'] for p in corpus],
                batch_size=64, normalize_embeddings=True, show_progress_bar=True)
            np.save(embeddings_path, self.embeddings)
        if self.embeddings.shape[0] != len(corpus):
            raise ValueError('Corpus and embeddings differ')
        metadata_path.write_text(json.dumps({'fingerprint': fingerprint, 'config': asdict(config),
            'encoder': encoder, 'corpus_size': len(corpus)}, indent=2))
        self._local = threading.local()
        self._encoder_lock = threading.Lock()

    @property
    def traces(self):
        if not hasattr(self._local, 'traces'):
            self._local.traces = []
        return self._local.traces

    @traces.setter
    def traces(self, value):
        self._local.traces = value

    def retrieve(self, query, top_k=5):
        candidates, trace, sources = self.graph.candidates(query)
        if not candidates:
            trace['selected_ids'] = []
            self.traces.append(trace)
            return []
        with self._encoder_lock:
            vector = self.encoder.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        # Dense ranking is strictly restricted to candidates produced by the hash graph.
        scores = self.embeddings[candidates] @ vector
        selected = np.lexsort((np.asarray(candidates), -scores))[:top_k]
        hits = [dict(self.corpus[candidates[j]], score=float(scores[j]),
                     sources=sorted(sources[candidates[j]])) for j in selected]
        trace['selected_ids'] = [p['id'] for p in hits]
        self.traces.append(trace)
        return hits
