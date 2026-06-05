from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from bayesian_fpde.bayesian_fpde import BayesianFPDEConfig, explain_bayesian_fpde
from bayesian_fpde.baselines import optional_baseline
from bayesian_fpde.datasets import encode_labels, fit_black_box, load_case_study_dataset, preprocess_train_test
from bayesian_fpde.fpde import FPDEConfig, class_prototypes, explain_fpde
from bayesian_fpde.plotting import save_ci_bar, save_rank_probability
from bayesian_fpde.utils import base_metadata, ensure_dirs, write_csv
from experiments.common import load_mode_config, parser_with_config


def main() -> int:
    args = parser_with_config("Run Bayesian-FPDE case studies.").parse_args()
    cfg = load_mode_config(args.config, args.mode)
    results_dir = Path(cfg.get("results_dir", "results"))
    figures_dir = Path(cfg.get("figures_dir", "figures"))
    ensure_dirs(results_dir, figures_dir)
    for name in cfg.get("datasets", ["breast_cancer", "wine"]):
        X_df, y_raw, dataset_name = load_case_study_dataset(str(name))
        y = encode_labels(y_raw)
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_df, y, test_size=0.25, random_state=int(cfg.get("seed", 0)), stratify=y)
        X_train, X_test, feature_names, _ = preprocess_train_test(X_train_raw.reset_index(drop=True), X_test_raw.reset_index(drop=True))
        model, model_name = fit_black_box(X_train, y_train, seed=int(cfg.get("seed", 0)), model_name=str(cfg.get("model", "random_forest")))
        prototypes, labels = class_prototypes(X_train, y_train)
        baseline = np.mean(X_train, axis=0)
        x = X_test[0]
        det_attr, target_label, _ = explain_fpde(model, x, prototypes, labels, config=FPDEConfig(mode="hyb", lambda_hyb=float(cfg.get("lambda_hyb", 0.5))), anchor=baseline)
        result = explain_bayesian_fpde(model, x, X_train, y_train, config=BayesianFPDEConfig(n_posterior_samples=int(cfg.get("posterior_samples", 200)), top_k=int(cfg.get("top_k", 5))), anchor=baseline, feature_names=feature_names, seed=int(cfg.get("seed", 0)))
        out = result.summary.copy()
        out["deterministic_fpde_score"] = det_attr
        for method in ["shap", "lime"]:
            base = optional_baseline(method, model, x, X_train, y_train, feature_names, target_label, seed=int(cfg.get("seed", 0)))
            out[f"{method}_score"] = base.attribution if base.attribution is not None else np.nan
            out[f"{method}_status"] = base.status
            out[f"{method}_error"] = base.error
        meta = base_metadata(dataset_name=dataset_name, method="bayesian_hyb_fpde", model=model_name, seed=int(cfg.get("seed", 0)), status="ok", error="")
        for key, value in meta.items():
            out[key] = value
        safe_name = str(dataset_name).replace(" ", "_").replace("/", "_")
        write_csv(out, results_dir / f"case_study_{safe_name}.csv")
        save_ci_bar(out, path=figures_dir / f"case_study_{safe_name}_credible_intervals.png", title=f"{dataset_name} credible intervals")
        save_rank_probability(out, path=figures_dir / f"case_study_{safe_name}_rank_probability.png", title=f"{dataset_name} rank probability")
        save_ci_bar(out.assign(posterior_mean=out["deterministic_fpde_score"], ci_lower_95=out["deterministic_fpde_score"], ci_upper_95=out["deterministic_fpde_score"]), path=figures_dir / f"case_study_{safe_name}_comparison.png", title=f"{dataset_name} deterministic comparison")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
