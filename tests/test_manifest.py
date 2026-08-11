import json, tempfile, unittest
from pathlib import Path
from finetune.data import ASRJsonlDataset

class TestManifest(unittest.TestCase):
    def test_missing_fields(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"x.jsonl"; p.write_text(json.dumps({"audio":"x.wav"})+"\n")
            with self.assertRaises(ValueError): ASRJsonlDataset(p)
