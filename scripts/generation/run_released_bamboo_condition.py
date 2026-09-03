#!/usr/bin/env python3
"""Sample/score one G0 condition using Bamboo-Mixer's native release stack."""

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import yaml


def is_finite(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bamboo-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-conductivity", type=float, required=True)
    parser.add_argument("--anion-ratio", type=float, required=True)
    parser.add_argument("--num-batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--temperature-c", type=float, default=25.0)
    parser.add_argument("--concentration", type=float, default=.1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    bamboo, output = Path(args.bamboo_root).resolve(), Path(args.output).resolve()
    native_dir, config_path = output.parent / "raw_bamboo_original", output.parent / "released_bamboo_condition.yaml"
    required = ("scripts/test_results/generate.py", "ckpts/generator/diffusion.pt",
                "ckpts/generator/decoder.pt", "ckpts/formula/optimal.pt",
                "ckpts/mono/optimal.pt", "run/solv_embs.pt", "run/salt_embs.pt",
                "config/test_results/generate_config.yaml")
    for relative in required:
        if not (bamboo / relative).is_file():
            raise FileNotFoundError(f"Required released Bamboo file is missing: {bamboo / relative}")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}; pass --overwrite")

    release_config = yaml.safe_load((bamboo / "config/test_results/generate_config.yaml").read_text())
    decoder_model = dict(release_config["decoder_config"]["model_config"])
    decoder_model["mol_dict"] = {
        "solv_dict_path": str(bamboo / "run/solv_embs.pt"),
        "salt_dict_path": str(bamboo / "run/salt_embs.pt"),
    }
    config = {
        "output_dir": str(native_dir), "num_batch": args.num_batches,
        "batch_size": args.batch_size, "temperature": args.temperature_c,
        "concentration": args.concentration,
        "conductivities": [args.target_conductivity], "anion_ratios": [args.anion_ratio],
        "diffusion_config": {"ckpt_path": str(bamboo / "ckpts/generator/diffusion.pt"),
                             "model_config": release_config["diffusion_config"]["model_config"]},
        "decoder_config": {"ckpt_path": str(bamboo / "ckpts/generator/decoder.pt"),
                           "model_config": decoder_model},
        "predictor_config": {
            "ckpt_path": str(bamboo / "ckpts/formula/optimal.pt"),
            "mono": {"ckpt_path": str(bamboo / "ckpts/mono/optimal.pt"),
                     "model_config": release_config["predictor_config"]["mono"]["model_config"]},
            "formula": release_config["predictor_config"]["formula"],
        },
    }
    native_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    # The public generator checkpoint contains two non-learnable SigmaScheduler
    # buffers from the later release.  The public Bamboo ``FormulaDiffusion``
    # class predates those buffers, so load it non-strictly while retaining all
    # learned diffusion weights and its original BetaScheduler sampling path.
    runner_path = native_dir / "run_native_bamboo_generate.py"
    runner_path.write_text(
        "import runpy, sys, torch\n"
        f"sys.path.insert(0, {str(bamboo)!r})\n"
        "from bamboo_mixer.generator import FormulaDiffusion\n"
        "from bamboo_mixer import predictor as predictor_module\n"
        "from scripts.generation.bamboo_released_predictor_compat import released_formula_predictor_class\n"
        "def compat_load(self, ckpt_path, device=None):\n"
        "    state = torch.load(ckpt_path, map_location=device, weights_only=True)['model_state_dict']\n"
        "    incompatible = self.load_state_dict(state, strict=False)\n"
        "    allowed = {'sigma_scheduler.sigmas', 'sigma_scheduler.sigmas_norm'}\n"
        "    if set(incompatible.unexpected_keys) - allowed or incompatible.missing_keys:\n"
        "        raise RuntimeError(f'Unexpected checkpoint incompatibility: {incompatible}')\n"
        "    if device is not None: self.to(device)\n"
        "FormulaDiffusion.load_ckpt = compat_load\n"
        "predictor_module.FormulaPredictor = released_formula_predictor_class()\n"
        f"sys.argv = ['generate.py', '--conf', {str(config_path)!r}]\n"
        f"runpy.run_path({str(bamboo / 'scripts/test_results/generate.py')!r}, run_name='__main__')\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(bamboo) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, str(runner_path)], cwd=bamboo, env=env, check=True)

    raw_path = native_dir / f"output_{args.target_conductivity}_{args.anion_ratio}.json"
    if not raw_path.is_file():
        raise FileNotFoundError(f"Expected Bamboo output is missing: {raw_path}")
    records = []
    for row in json.loads(raw_path.read_text()):
        conductivity, ratio = row.get("conductivity"), row.get("anion_ratio")
        conductivity_ok = is_finite(conductivity) and 0.0 <= float(conductivity) <= 100.0
        ratio_ok = is_finite(ratio) and 0.0 <= float(ratio) <= 1.0
        trusted = conductivity_ok and ratio_ok
        conductivity = float(conductivity) if conductivity_ok else None
        ratio = float(ratio) if ratio_ok else None
        entry = {
            "bow": row["bow"], "temperature_c": args.temperature_c,
            "salt_molar_ratio": args.concentration,
            "target_conductivity_ms_cm": args.target_conductivity,
            "anion_ratio_condition": args.anion_ratio, "anion_ratio_pred": ratio,
            "anion_ratio_eligibility_min": 0.0, "anion_ratio_eligibility_max": 1.0,
            "within_anion_ratio_eligibility": ratio_ok,
            "oracle_conductivity_pred_ms_cm": conductivity,
            "conductivity_pred_ms_cm": conductivity if trusted else None,
            "prediction_status": "trusted" if trusted else "out_of_domain",
            "explanation_residual_25c": 0.0, "selection_policy": "joint_target",
            "oracle_source": "released_bamboo_formula_predictor",
        }
        if trusted:
            conductivity_error = abs(conductivity - args.target_conductivity)
            anion_error = abs(ratio - args.anion_ratio)
            entry.update({"conductivity_target_abs_error_ms_cm": conductivity_error,
                          "anion_ratio_abs_error": anion_error,
                          "candidate_rank_score": -(conductivity_error / 2.0 + anion_error / .1),
                          "eligibility_reason": ""})
        else:
            entry.update({"candidate_rank_score": None,
                          "eligibility_reason": "nonfinite_or_out_of_domain_bamboo_prediction"})
        records.append(entry)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2) + "\n")
    (output.parent / "released_bamboo_provenance.json").write_text(json.dumps({
        "generator": str(bamboo / "ckpts/generator"),
        "formula_predictor": str(bamboo / "ckpts/formula/optimal.pt"),
        "mono_predictor": str(bamboo / "ckpts/mono/optimal.pt"),
        "native_config": str(config_path), "native_output": str(raw_path),
        "scoring": "released Bamboo-Mixer predictor, not current dual-task oracle",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
