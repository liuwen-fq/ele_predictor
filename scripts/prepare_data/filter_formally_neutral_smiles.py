#!/usr/bin/env python3
"""Split a SMILES CSV into net-neutral and strictly uncharged molecules.

Formal charges in SMILES occur inside bracketed atom expressions, for example
``[O-]``, ``[N+]``, ``[Fe+2]`` or ``[N--]``.  The script deliberately reports
both definitions of neutral:

* net neutral: the sum of atomic formal charges is zero (may be zwitterionic)
* strictly uncharged: no atom has a non-zero formal charge
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


BRACKET_ATOM = re.compile(r"\[([^\]]+)\]")
CHARGE = re.compile(r"([+-])(\d*)")


def formal_charge(smiles: str) -> tuple[int, bool]:
    """Return (net formal charge, whether any atom is formally charged)."""
    net_charge = 0
    charged_atom = False
    for atom in BRACKET_ATOM.findall(smiles):
        atom_charge = 0
        for sign, magnitude in CHARGE.findall(atom):
            amount = int(magnitude) if magnitude else 1
            atom_charge += amount if sign == "+" else -amount
        net_charge += atom_charge
        charged_atom |= atom_charge != 0
    return net_charge, charged_atom


def write_row(writer: csv.DictWriter, row: dict[str, str], net_charge: int) -> None:
    writer.writerow(
        {
            "molecule_id": row["molecule_id"],
            "smiles": row["smiles"],
            "formal_charge": net_charge,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Input CSV with molecule_id,smiles")
    parser.add_argument("net_neutral", type=Path, help="Output CSV: net formal charge = 0")
    parser.add_argument("strictly_uncharged", type=Path, help="Output CSV: no charged atoms")
    args = parser.parse_args()

    fieldnames = ["molecule_id", "smiles", "formal_charge"]
    net_count = strict_count = non_neutral_count = 0
    with (
        args.source.open(newline="", encoding="utf-8") as source,
        args.net_neutral.open("w", newline="", encoding="utf-8") as net_file,
        args.strictly_uncharged.open("w", newline="", encoding="utf-8") as strict_file,
    ):
        reader = csv.DictReader(source)
        net_writer = csv.DictWriter(net_file, fieldnames=fieldnames)
        strict_writer = csv.DictWriter(strict_file, fieldnames=fieldnames)
        net_writer.writeheader()
        strict_writer.writeheader()
        for row in reader:
            net_charge, charged_atom = formal_charge(row["smiles"])
            if net_charge == 0:
                write_row(net_writer, row, net_charge)
                net_count += 1
                if not charged_atom:
                    write_row(strict_writer, row, net_charge)
                    strict_count += 1
            else:
                non_neutral_count += 1

    print(f"Net-neutral molecules: {net_count:,}")
    print(f"Strictly uncharged molecules: {strict_count:,}")
    print(f"Non-neutral molecules excluded: {non_neutral_count:,}")


if __name__ == "__main__":
    main()
