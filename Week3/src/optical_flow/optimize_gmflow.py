import json
import optuna
from pathlib import Path

from src.optical_flow.runner import run_sequence
from src.optical_flow.gmflow_method import get_gmflow_default_model_config


RESULTS_DIR = Path("./results/gmflow")
RESULTS_DIR.mkdir(exist_ok=True)
SEQ = 45


def objective(trial):
    params = get_gmflow_default_model_config()

    attn_splits = trial.suggest_categorical("attn_splits", [1, 2, 4])
    corr_radius = trial.suggest_categorical("corr_radius", [-1, 4, 8])
    prop_radius = trial.suggest_categorical("prop_radius", [-1, 1, 2])

    params["attn_splits_list"] = [attn_splits]
    params["corr_radius_list"] = [corr_radius]
    params["prop_radius_list"] = [prop_radius]

    # safe padding for the chosen split so it does not break
    params["padding_factor"] = max(16, 8 * attn_splits)

    flow, msen, pepn, info = run_sequence(
        seq=SEQ,
        method="gmflow",
        method_params=params,
    )

    return msen, pepn


if __name__ == "__main__":
    study = optuna.create_study(study_name="GMFlow optimization", directions=["minimize", "minimize"])
    study.optimize(objective, n_trials=20)

    print(f"Number of trials: {len(study.trials)}")
    print(f"Number of Pareto-optimal trials: {len(study.best_trials)}")

    pareto_trials = []
    for t in study.best_trials:
        pareto_trials.append({
            "trial_number": t.number,
            "msen": t.values[0],
            "pepn": t.values[1],
            "params": t.params,
        })

    df = study.trials_dataframe()
    df.to_csv(RESULTS_DIR / "optuna_gmflow_trials.csv", index=False)

    with open(RESULTS_DIR / "optuna_gmflow_pareto.json", "w") as f:
        json.dump(pareto_trials, f, indent=2)