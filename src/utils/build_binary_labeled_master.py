"""
Build a binary-view JSONL from labeled_master.jsonl.

Rules:
- Fakeddit records use original_label as source of truth:
  - True -> REAL
  - Fake -> FAKE
- Reddit records use the current 6-class label:
  - TRUE, MOSTLY_TRUE, HALF_TRUE -> REAL
  - BARELY_TRUE, FALSE, PANTS_ON_FIRE -> FAKE

The script preserves all original fields and adds:
- label_binary
- label_binary_source
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.data.label_utils import derive_binary_label


def build_binary_view(input_path: Path, output_path: Path, replace_label: bool = False) -> None:
    source_counter = Counter()
    binary_counter = Counter()
    strategy_counter = Counter()
    total = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as f_in, output_path.open("w", encoding="utf-8") as f_out:
        for line_number, line in enumerate(f_in, start=1):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            total += 1

            binary_label, strategy = derive_binary_label(record)
            record["label_binary"] = binary_label
            record["label_binary_source"] = strategy

            if replace_label:
                current_label = str(record.get("label", "")).strip()
                if current_label and current_label != binary_label and "label_6class" not in record:
                    record["label_6class"] = current_label
                record["label"] = binary_label

            source_counter[str(record.get("source_dataset", "MISSING"))] += 1
            binary_counter[binary_label] += 1
            strategy_counter[strategy] += 1

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("=" * 60)
    print("BINARY VIEW BUILD COMPLETE")
    print("=" * 60)
    print(f"Input:   {input_path}")
    print(f"Output:  {output_path}")
    print(f"Total:   {total}")
    print(f"Sources: {dict(source_counter)}")
    print(f"Binary:  {dict(binary_counter)}")
    print(f"Rules:   {dict(strategy_counter)}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Build binary-view labeled_master.jsonl")
    parser.add_argument(
        "--input",
        default="data/03_clean/Fakeddit/labeled_master.jsonl",
        help="Path to source labeled_master.jsonl",
    )
    parser.add_argument(
        "--output",
        default="data/03_clean/Fakeddit/labeled_master_binary.jsonl",
        help="Path to output binary JSONL",
    )
    parser.add_argument(
        "--replace-label",
        action="store_true",
        help="Overwrite the output record label field with REAL/FAKE and preserve old 6-class label in label_6class",
    )
    args = parser.parse_args()

    build_binary_view(Path(args.input), Path(args.output), replace_label=args.replace_label)


if __name__ == "__main__":
    main()
