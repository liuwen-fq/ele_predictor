"""Evaluate predicted experimental conductivity in mS/cm."""

import argparse
import json
import math

LOG_SHIFT = 4.6002
LOG_SCALE = 8.3344


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",
                        required=True,
                        help="JSON written by scripts/test_results/predict.py")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--prediction-key",
        default="conductivity_pred",
        help="Physical-unit prediction field to evaluate",
    )
    args = parser.parse_args()

    with open(args.input, "r") as file:
        records = json.load(file)

    targets = []
    predictions = []
    for record in records:
        if not record.get("conductivity_mask", False):
            continue
        if args.prediction_key not in record:
            raise ValueError(
                f"prediction JSON has no {args.prediction_key} field")
        target = math.exp(record["conductivity"] * LOG_SCALE - LOG_SHIFT)
        targets.append(target)
        predictions.append(float(record[args.prediction_key]))

    if not targets:
        raise ValueError("no experimental conductivity labels found")
    mean_target = sum(targets) / len(targets)
    errors = [
        prediction - target
        for prediction, target in zip(predictions, targets)
    ]
    mae = sum(abs(error) for error in errors) / len(errors)
    rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
    ss_res = sum(error * error for error in errors)
    ss_tot = sum((target - mean_target)**2 for target in targets)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    log_targets = [math.log(target) for target in targets]
    log_predictions = [
        math.log(max(prediction, 1e-12)) for prediction in predictions
    ]
    log_mae = sum(
        abs(prediction - target)
        for prediction, target in zip(log_predictions, log_targets))
    log_mae /= len(targets)

    metrics = {
        "prediction_key": args.prediction_key,
        "num_experimental_samples": len(targets),
        "conductivity_mae_mS_cm": mae,
        "conductivity_rmse_mS_cm": rmse,
        "conductivity_r2": r2,
        "log_conductivity_mae": log_mae,
        "normalization": {
            "target_mS_cm": "exp(normalized * 8.3344 - 4.6002)"
        },
    }
    with open(args.output, "w") as file:
        json.dump(metrics, file, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
