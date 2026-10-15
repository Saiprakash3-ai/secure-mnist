"""Collect results/*.json into results/summary.csv for the report."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from common import RESULT_DIR


def main() -> None:
    rows = []
    for path in sorted(RESULT_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if "accuracy" not in data:
            continue
        rows.append({
            "tag": data.get("tag", path.stem),
            "accuracy": round(data.get("accuracy", 0.0), 4),
            "loss": round(data.get("loss", 0.0), 4),
            "n": data.get("num_samples", ""),
            "inference_ms_per_sample": round(data.get("inference_time_per_sample_ms", 0.0), 4),
        })

    out = RESULT_DIR / "summary.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["tag", "accuracy", "loss", "n", "inference_ms_per_sample"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[summary] {len(rows)} runs -> {out}")
    for r in rows:
        print(f"  {r['tag']:<38} acc={r['accuracy']:.4f} loss={r['loss']:.4f}")


if __name__ == "__main__":
    main()
