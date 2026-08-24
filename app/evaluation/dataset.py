import json
from pathlib import Path
from typing import List, Dict


class EvaluationDataset:

    def __init__(self, path: str):
        self.path = Path(path)

    def load(self) -> List[Dict]:
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)
