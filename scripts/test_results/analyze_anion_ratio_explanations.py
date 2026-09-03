#!/usr/bin/env python3
"""Export anion explanation summaries only after the MAE gate passes."""

import argparse
import csv
import heapq
import json
from collections import defaultdict
from pathlib import Path


def explanation_records(value, field_name):
    """Normalize current dict output and the older list output to a list."""
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        invalid = [item for item in value if not isinstance(item, dict)]
        if invalid:
            raise TypeError(
                f"{field_name} must contain explanation objects, got "
                f"{type(invalid[0]).__name__}")
        return value
    raise TypeError(
        f"{field_name} must be an explanation object or list, got "
        f"{type(value).__name__}")


def iter_json_array(path, chunk_size=1 << 20):
    """Yield a top-level JSON array one item at a time.

    Prediction exports include full fragment diagnostics and can exceed several
    GB.  Loading the entire array would make the post-processing analysis use
    substantially more memory than prediction itself.
    """
    decoder = json.JSONDecoder()
    with Path(path).open("r", encoding="utf-8") as handle:
        buffer, eof = "", False

        def read_more():
            nonlocal buffer, eof
            chunk = handle.read(chunk_size)
            if chunk:
                buffer += chunk
            else:
                eof = True

        read_more()
        while True:
            buffer = buffer.lstrip()
            if not buffer:
                if eof:
                    raise ValueError("prediction JSON ended before the top-level array")
                read_more()
                continue
            if buffer[0] != "[":
                raise ValueError("prediction JSON must be a top-level array")
            buffer = buffer[1:]
            break

        while True:
            buffer = buffer.lstrip()
            if not buffer:
                if eof:
                    raise ValueError("prediction JSON ended inside the top-level array")
                read_more()
                continue
            if buffer[0] == "]":
                return
            if buffer[0] == ",":
                buffer = buffer[1:]
                continue
            try:
                record, offset = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                if eof:
                    raise ValueError("invalid or truncated prediction JSON") from None
                read_more()
                continue
            if not isinstance(record, dict):
                raise TypeError(
                    "prediction JSON must contain formula objects, got "
                    f"{type(record).__name__}")
            yield record
            buffer = buffer[offset:]


def update_stat(stats, key, value):
    stats[key]["n"] += 1
    stats[key]["sum"] += float(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-interactions", type=int, default=10000,
                        help="retain this many globally strongest pair effects")
    args = parser.parse_args()
    if args.max_interactions <= 0:
        raise ValueError("--max-interactions must be positive")
    metric = json.loads(Path(args.metrics).read_text())
    if not metric.get("analysis_eligible"):
        raise SystemExit(
            "anion explanation analysis withheld: explain-head MAE exceeds "
            "the permitted degradation; inspect metrics first")
    fragments = defaultdict(lambda: {"n": 0, "sum": 0.0})
    properties = defaultdict(lambda: {"n": 0, "sum": 0.0})
    interactions = []
    pair_counter, total_pairs = 0, 0
    processed, eligible = 0, 0
    for row in iter_json_array(args.input):
        processed += 1
        if processed % 1000 == 0:
            print(f"Processed {processed} prediction records", flush=True)
        if not row.get("anion_ratio_mask", False):
            continue
        eligible += 1
        for formula in explanation_records(
                row.get("anion_ratio_fragment_explanations"),
                "anion_ratio_fragment_explanations"):
            for pair in formula.get("top_fragment_pairs", []):
                effect = float(pair["effect_logit"])
                pair_counter += 1
                total_pairs += 1
                candidate = (abs(effect), pair_counter, {
                    "effect_logit": effect,
                    "left": pair["left"],
                    "right": pair["right"],
                })
                if len(interactions) < args.max_interactions:
                    heapq.heappush(interactions, candidate)
                elif candidate[0] > interactions[0][0]:
                    heapq.heapreplace(interactions, candidate)
        for molecule in row.get("fragment_attention", {}).get("solvents", []):
            for fragment in molecule.get("fragments", []):
                if "anion_ratio_main_effect_logit" in fragment:
                    update_stat(fragments, fragment.get("smiles", ""),
                                fragment["anion_ratio_main_effect_logit"])
        for formula in explanation_records(
                row.get("anion_ratio_property_explanations"),
                "anion_ratio_property_explanations"):
            for molecule in formula.get("solvent_property_contributions", []):
                for name, value in molecule.get("property_effects_normalized", {}).items():
                    update_stat(properties, name, value)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    def write(name, rows, fields):
        with (out / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    write("anion_fragment_contributions.csv", [
        {"fragment_smiles": key, "n": value["n"],
         "mean_logit_effect": value["sum"] / value["n"]}
        for key, value in sorted(fragments.items(),
                                 key=lambda item: -abs(item[1]["sum"] / item[1]["n"]))],
        ["fragment_smiles", "n", "mean_logit_effect"])
    write("anion_property_contributions.csv", [
        {"property": key, "n": value["n"],
         "mean_logit_effect": value["sum"] / value["n"]}
        for key, value in sorted(properties.items())],
        ["property", "n", "mean_logit_effect"])
    interaction_rows = [record for _, _, record in sorted(
        interactions, key=lambda item: item[0], reverse=True)]
    (out / "anion_interactions.json").write_text(
        json.dumps(interaction_rows, indent=2) + "\n")
    (out / "analysis_summary.json").write_text(json.dumps({
        "prediction_records": processed,
        "anion_ratio_labelled_records": eligible,
        "fragment_types": len(fragments),
        "property_types": len(properties),
        "fragment_pair_candidates": total_pairs,
        "retained_strongest_fragment_pairs": len(interaction_rows),
    }, indent=2) + "\n")
    (out / "analysis_provenance.json").write_text(json.dumps(metric, indent=2) + "\n")


if __name__ == "__main__":
    main()
