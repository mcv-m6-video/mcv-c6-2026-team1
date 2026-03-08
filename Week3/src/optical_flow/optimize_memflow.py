import sys
import json
import optuna
from pathlib import Path

from src.optical_flow.runner import run_sequence


RESULTS_DIR = Path("./results/memflow")
RESULTS_DIR.mkdir(exist_ok=True)
SEQ = 45


# Short trial
def objective(trial):
    model_family = trial.suggest_categorical("model_family", ["MemFlowNet", "MemFlowNet_T"])
    stage = trial.suggest_categorical("stage", ["kitti", "sintel"])

    if model_family == "MemFlowNet":
        if stage == "kitti":
            method = "memflow_kitti"
        elif stage == "sintel":
            method = "memflow_sintel"
    else:
        if stage == "kitti":
            method = "memflow_t_kitti"
        elif stage == "sintel":
            method = "memflow_t_kitti"

    flow, msen, pepn, info = run_sequence(
        seq=SEQ,
        method=method,
    )

    return msen, pepn


if __name__ == "__main__":
    study = optuna.create_study(study_name="MemFlow optimization", directions=["minimize", "minimize"])
    study.optimize(objective, n_trials=10)

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
    df.to_csv(RESULTS_DIR / "optuna_memflow_trials.csv", index=False)

    with open(RESULTS_DIR / "optuna_memflow_pareto.json", "w") as f:
        json.dump(pareto_trials, f, indent=2)