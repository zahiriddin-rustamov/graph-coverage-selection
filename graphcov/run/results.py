"""Shared result-saving utilities for experiment scripts.

Used by experiment.py, compare_k.py, and future comparison scripts.
"""

import csv
import json
import subprocess
import sys
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / 'results'


def generate_run_id() -> str:
    """Generate a timestamp-based run ID."""
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def create_run_dir(run_id: str, base_dir: Optional[Path] = None) -> Path:
    """Create and return a run directory."""
    if base_dir is None:
        base_dir = DEFAULT_OUTPUT_DIR
    run_dir = Path(base_dir) / 'runs' / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def get_git_commit() -> Optional[str]:
    """Get current git commit hash (short)."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).parent.parent.parent
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return None


def save_config(run_dir: Path, config: Dict[str, Any]):
    """Save experiment config as JSON."""
    # Convert non-serializable types
    serializable = {}
    for k, v in config.items():
        if isinstance(v, Path):
            serializable[k] = str(v)
        elif isinstance(v, np.integer):
            serializable[k] = int(v)
        elif isinstance(v, np.floating):
            serializable[k] = float(v)
        elif isinstance(v, np.ndarray):
            serializable[k] = v.tolist()
        else:
            serializable[k] = v

    with open(run_dir / 'config.json', 'w') as f:
        json.dump(serializable, f, indent=2)


def save_summary(run_dir: Path, summary: Dict[str, Any]):
    """Save run summary as JSON."""
    with open(run_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


def append_to_csv(filepath: Path, row: Dict[str, Any]):
    """Append a single row to CSV file, creating with headers if new.

    If the row has new columns not in the existing header, rewrites
    the file with the expanded header to keep the CSV well-formed.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if not filepath.exists() or filepath.stat().st_size == 0:
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        return

    # Read existing header
    with open(filepath, 'r', newline='') as f:
        reader = csv.reader(f)
        existing_header = next(reader)

    new_keys = [k for k in row.keys() if k not in existing_header]

    if not new_keys:
        # Simple append — no schema change
        with open(filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=existing_header, extrasaction='ignore')
            writer.writerow(row)
    else:
        # Schema changed — rewrite with expanded header
        expanded_header = existing_header + new_keys
        with open(filepath, 'r', newline='') as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=expanded_header)
            writer.writeheader()
            writer.writerows(existing_rows)
            writer.writerow(row)


def save_training_history(run_dir: Path, rows: List[Dict[str, Any]],
                          filename: str = 'training_history.csv'):
    """Save per-checkpoint training metrics as CSV."""
    if not rows:
        return
    filepath = run_dir / filename
    # Collect all keys across all rows (some rows may have extra fields like test_acc)
    fieldnames = list(dict.fromkeys(k for row in rows for k in row.keys()))
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def save_selections(run_dir: Path, selections: Dict[str, List[int]],
                    filename: str = 'selections.json'):
    """Save selected indices as JSON. Keys can be k values, sizes, etc."""
    # Convert keys to strings for JSON
    serializable = {str(k): [int(x) for x in sorted(v)] for k, v in selections.items()}
    with open(run_dir / filename, 'w') as f:
        json.dump(serializable, f)
