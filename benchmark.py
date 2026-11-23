"""Resumable HotpotQA benchmark; references are used only for evaluation."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import fcntl
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from evaluate import evaluate_answers
from main import GraphRetriever, LLM, answer_question, load_corpus
from retrieve import HashConfig


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def retrieval_statistics(rows, corpus_size):
    sizes = np.asarray([trace['candidate_count'] for row in rows for trace in row['retrieval']])
    if not len(sizes):
        return {'retrieval_calls': 0, 'corpus_size': corpus_size}
    result = {'retrieval_calls': len(sizes), 'corpus_size': corpus_size}
    for name, value in [('mean', sizes.mean()), ('median', np.median(sizes)),
                        ('min', sizes.min()), ('max', sizes.max()),
                        ('p95', np.percentile(sizes, 95))]:
        result['candidate_' + name] = float(value)
        result[name + '_corpus_percent'] = float(value / corpus_size * 100)
    return result


def run_question(index, item, run, retriever, llm, top_k):
    output = run / f'{index:04d}.json'
    row = json.loads(output.read_text()) if output.exists() else None
    if row and (row['id'] != str(item['_id']) or row['question'] != item['question']):
        raise ValueError('Checkpoint question differs')
    for attempt in range(3):
        try:
            if row is None:
                retriever.traces = []
                before = time.monotonic()
                # No reference answer, per-question context or supporting facts enter generation.
                result = answer_question(item['question'], retriever, llm, top_k)
                row = result | {'id': str(item['_id']), 'index': index,
                    'correct_answer': item['answer'], 'retrieval': retriever.traces,
                    'answer_seconds': time.monotonic() - before}
                atomic_json(output, row)
            if 'evaluation' not in row:
                row['evaluation'] = evaluate_answers([row['question']], [row['answer']],
                    [row['correct_answer']], 'both', llm)
                atomic_json(output, row)
            return row
        except Exception as error:
            print(json.dumps({'event': 'retry', 'index': index, 'attempt': attempt + 1,
                              'error_type': type(error).__name__}), flush=True)
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='dataset/hotpotqa.json')
    parser.add_argument('--corpus', default='dataset/hotpotqa_corpus.json')
    parser.add_argument('--encoder', default='all-MiniLM-L6-v2')
    parser.add_argument('--state', default='state')
    parser.add_argument('--run', default='runs/hotpotqa_200')
    parser.add_argument('--limit', type=int, default=200)
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--model', default='deepseek-flash')
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError('workers must be positive')
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    run_lock = (run / '.lock').open('w')
    fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    dataset = json.loads(Path(args.dataset).read_text())[:args.limit]
    if len(dataset) != args.limit:
        raise ValueError('Dataset is shorter than the requested benchmark')
    config = {k: v for k, v in vars(args).items() if k != 'workers'} | {
        'hash_config': asdict(HashConfig()),
        'dataset_sha256': hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
        'corpus_sha256': hashlib.sha256(Path(args.corpus).read_bytes()).hexdigest()}
    if (run / 'config.json').exists():
        if json.loads((run / 'config.json').read_text()) != config:
            raise ValueError('Run configuration changed; use a different output directory')
    else:
        atomic_json(run / 'config.json', config)
    retriever = GraphRetriever(load_corpus(args.corpus), args.state, args.encoder)
    llm, completed = LLM(args.model), []
    started = time.monotonic()
    # Graph and embeddings are immutable. Questions may run concurrently, while
    # the decomposition and retrieval steps within each question remain sequential.
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_question, index, item, run, retriever, llm, args.top_k)
                   for index, item in enumerate(dataset)]
        for future in as_completed(futures):
            row = future.result()
            completed.append(row)
            scores = {metric: sum(r['evaluation']['scores'][metric] for r in completed) / len(completed)
                      for metric in ('ContainsMatch', 'LLMAccuracy')}
            summary = {'completed': len(completed), 'requested': args.limit, 'scores': scores,
                'correct': {metric: round(score * len(completed)) for metric, score in scores.items()},
                'graph': retriever.graph.stats, 'workers': args.workers,
                'retrieval': retrieval_statistics(completed, retriever.graph.stats['chunks']),
                'elapsed_seconds_this_process': round(time.monotonic() - started, 2)}
            atomic_json(run / 'summary.json', summary)
            print(json.dumps(summary | {'index': row['index']}), flush=True)
    with (run / 'predictions.jsonl').open('w') as handle:
        for row in sorted(completed, key=lambda r: r['index']):
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    print('COMPLETE ' + json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
