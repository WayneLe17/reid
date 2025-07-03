import json
from pathlib import Path
from typing import Dict, Optional

class TrackingResultsConverter:
    @staticmethod
    def load_tracking_data(results_path: str) -> Optional[Dict]:
        results_path = Path(results_path)
        
        if results_path.is_file() and results_path.suffix == '.json':
            with open(results_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        elif results_path.is_dir():
            unified_file = results_path / "tracking_results.json"
            if unified_file.exists():
                with open(unified_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        
        return None