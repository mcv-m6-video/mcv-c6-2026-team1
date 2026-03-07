import json
import optuna
from pathlib import Path

from src.optical_flow.runner import run_sequence
from src.optical_flow.pyflow_method import get_pyflow_default_params


RESULTS_DIR = Path("./results/pyflow")
RESULTS_DIR.mkdir(exist_ok=True)
SEQ = 45


def objective(trial):
    params = get_pyflow_default_params()

    params["alpha"] = trial.suggest_float("alpha", 0.001, 0.05, log=True)
    params["ratio"] = trial.suggest_float("ratio", 0.5, 0.95)
    params["minWidth"] = trial.suggest_int("minWidth", 10, 40)
    params["nOuterFPIterations"] = trial.suggest_int("nOuterFPIterations", 3, 10)
    params["nSORIterations"] = trial.suggest_int("nSORIterations", 10, 50)

    flow, msen, pepn, info = run_sequence(
        seq=SEQ,
        method="pyflow",
        method_params=params,
    )

    return msen, pepn


if __name__ == "__main__":
    study = optuna.create_study(study_name="PyFlow optimization", directions=["minimize", "minimize"])
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
    df.to_csv(RESULTS_DIR / "optuna_pyflow_trials.csv", index=False)

    with open(RESULTS_DIR / "optuna_pyflow_pareto.json", "w") as f:
        json.dump(pareto_trials, f, indent=2)