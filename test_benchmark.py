import unittest

from benchmark import retrieval_statistics


class RetrievalStatisticsTests(unittest.TestCase):
    def test_counts_are_per_retrieval_with_corpus_percentages(self):
        rows = [{'retrieval': [{'candidate_count': 100}, {'candidate_count': 200}]},
                {'retrieval': [{'candidate_count': 300}]}]
        result = retrieval_statistics(rows, 2000)
        self.assertEqual(result['retrieval_calls'], 3)
        self.assertEqual(result['candidate_mean'], 200)
        self.assertEqual(result['candidate_min'], 100)
        self.assertEqual(result['candidate_max'], 300)
        self.assertEqual(result['candidate_median'], 200)
        self.assertEqual(result['candidate_p95'], 290)
        self.assertEqual(result['mean_corpus_percent'], 10)
        self.assertEqual(result['min_corpus_percent'], 5)
        self.assertEqual(result['max_corpus_percent'], 15)

    def test_empty_run_has_no_retrieval_calls(self):
        self.assertEqual(retrieval_statistics([], 2000),
                         {'retrieval_calls': 0, 'corpus_size': 2000})


if __name__ == '__main__':
    unittest.main()
