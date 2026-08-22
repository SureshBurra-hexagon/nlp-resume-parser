from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.resume_parser import parse_resume_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse a resume file (TXT/HTML/DOCX/PDF)")
    parser.add_argument("--file", required=True, help="Path to resume file")
    args = parser.parse_args()

    parsed = parse_resume_file(args.file)
    print(json.dumps(parsed, indent=2))


if __name__ == "__main__":
    main()
