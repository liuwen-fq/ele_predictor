#!/usr/bin/env python3
"""Export the SMILES column from the column-oriented GDB17 JSON dataset.

The source is ~15 GB, but the SMILES section is modest in size.  This script
locates that section without loading the full source file and exports a CSV
whose zero-based ``molecule_id`` is the index used by every other GDB17 field.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


MARKER = b'"SMILES": ['
CHUNK_SIZE = 1024 * 1024


def find_smiles_offset(source: Path) -> int:
    """Return the byte offset immediately after the opening SMILES bracket."""
    overlap = b""
    scanned = 0
    with source.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            data = overlap + chunk
            found = data.find(MARKER)
            if found >= 0:
                return scanned - len(overlap) + found + len(MARKER)
            overlap = data[-(len(MARKER) - 1) :]
            scanned += len(chunk)
    raise ValueError(f"Could not find {MARKER!r} in {source}")


def iter_smiles(source: Path):
    """Yield JSON strings in the SMILES array using bounded memory."""
    with source.open("rb") as handle:
        handle.seek(find_smiles_offset(source))
        in_string = False
        escaped = False
        token = bytearray()
        while byte := handle.read(1):
            char = byte[0]
            if not in_string:
                if char == ord('"'):
                    in_string = True
                    token = bytearray(byte)
                elif char == ord(']'):
                    return
                continue

            token.append(char)
            if escaped:
                escaped = False
            elif char == ord('\\'):
                escaped = True
            elif char == ord('"'):
                yield json.loads(token)
                in_string = False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Path to GDB17.json")
    parser.add_argument("output", type=Path, help="CSV destination")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["molecule_id", "smiles"])
        for molecule_id, smiles in enumerate(iter_smiles(args.source)):
            writer.writerow([molecule_id, smiles])
            count += 1
    print(f"Exported {count:,} SMILES to {args.output}")


if __name__ == "__main__":
    main()
