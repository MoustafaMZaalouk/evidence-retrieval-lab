import unittest
import tempfile
import json
from pathlib import Path
from demo import run

class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.output=Path(cls.temp.name)/'results.json'
        cls.r=run(cls.output)
    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()
    def test_output_matches_returned_result(self):
        self.assertEqual(json.loads(self.output.read_text()),self.r)
    def test_rank_bounds(self):
        self.assertTrue(all(1<=r['rank']<=self.r['documents'] for r in self.r['rankings']))
    def test_recall_monotonic(self):
        self.assertGreaterEqual(self.r['recall_at_3'],self.r['recall_at_1'])
    def test_unknown_abstention(self):
        self.assertTrue(all(r['abstained'] for r in self.r['unknown_results']))

if __name__ == "__main__":
    unittest.main()
