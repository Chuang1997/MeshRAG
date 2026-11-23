import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import unittest
import numpy as np

from retrieve import GraphRetriever, HashConfig, HashGraph, MinHasher, extract_entities, merge_communities


class HashTests(unittest.TestCase):
    def test_query_case_and_hash_determinism(self):
        config = HashConfig()
        first, second = MinHasher(config), MinHasher(config)
        a = extract_entities('who directed casablanca?')
        b = extract_entities('Who directed Casablanca?')
        self.assertEqual(a, b)
        np.testing.assert_array_equal(first.signature(a), second.signature(b))
        np.testing.assert_array_equal(first.signature({'film', 'actor'}), first.signature({'actor', 'film'}))

    def test_overlapping_buckets_merge_transitively(self):
        raw = [{0, 1, 2}, {1, 2, 3}, {2, 3, 4}, {4, 5}]
        merged = merge_communities(raw, 0.6, 10)
        self.assertIn([0, 1, 2, 3, 4], merged)
        self.assertIn([4, 5], merged)
        self.assertEqual(merged, merge_communities(raw[::-1], 0.6, 10))

    def test_graph_roundtrip_and_candidate_union(self):
        corpus = [{'id': str(i), 'text': text} for i, text in enumerate([
            'Casablanca film directed Michael Curtiz starring Humphrey Bogart',
            'Casablanca film directed Michael Curtiz starring Ingrid Bergman',
            'Humphrey Bogart actor starred Casablanca film',
            'Volcano lava geology mountain eruption'])]
        graph = HashGraph(corpus).build()
        self.assertTrue(graph.edges)
        for edge in graph.edges:
            a, b = edge['source'], edge['target']
            self.assertTrue(edge['common_entities'])
            self.assertEqual(set(edge['common_entities']), extract_entities(corpus[a]['text']) & extract_entities(corpus[b]['text']))
            self.assertTrue(0 <= edge['hash_similarity'] <= 1)
            self.assertIn(a, graph.neighbors[b])
        result = graph.candidates('who directed casablanca')
        self.assertTrue(any('community' in origins for origins in result[2].values()))
        self.assertTrue(any('chunk_neighbor' in origins for origins in result[2].values()))
        with tempfile.TemporaryDirectory() as folder:
            graph.save(folder)
            loaded = HashGraph(corpus).load(folder)
            self.assertEqual(result, loaded.candidates('who directed casablanca'))
        self.assertEqual(graph.candidates('the and or')[0], [])

    def test_reranking_stays_inside_candidates_and_isolates_thread_traces(self):
        retriever = GraphRetriever.__new__(GraphRetriever)
        retriever.corpus = [{'id': str(i), 'text': str(i)} for i in range(3)]
        retriever.embeddings = np.array([[0., 1.], [0.8, 0.6], [1., 0.]])
        retriever.encoder = SimpleNamespace(encode=lambda *a, **kw: np.array([[1., 0.]]))
        retriever.graph = SimpleNamespace(candidates=lambda query: (
            [0, 1], {'query': query}, {0: {'hash_nearest'}, 1: {'community'}}))
        retriever._local = threading.local()
        retriever._encoder_lock = threading.Lock()
        barrier = threading.Barrier(2)
        def run(query):
            retriever.traces = []
            hits = retriever.retrieve(query, 1)
            barrier.wait(timeout=5)
            return hits, retriever.traces
        with ThreadPoolExecutor(2) as executor:
            results = list(executor.map(run, ['first query', 'second query']))
        for query, (hits, traces) in zip(['first query', 'second query'], results):
            self.assertEqual([h['id'] for h in hits], ['1'])
            self.assertEqual([t['query'] for t in traces], [query])


if __name__ == '__main__':
    unittest.main()
