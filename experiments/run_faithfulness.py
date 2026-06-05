from __future__ import annotations

from pathlib import Path

import pandas as pd

from bayesian_fpde.plotting import save_line_plot
from bayesian_fpde.utils import ensure_dirs, setup_logging, write_csv
from experiments.common import evaluate_methods_for_dataset, load_mode_config, load_tabular_openml_or_local, parser_with_config


def main() -> int:
    args = parser_with_config("Run deletion/insertion faithfulness experiments.").parse_args()
    cfg = load_mode_config(args.config, args.mode)
    logger = setup_logging(cfg.get("logs_dir", "logs"), "faithfulness")
    results_dir = Path(cfg.get("results_dir", "results"))
    figures_dir = Path(cfg.get("figures_dir", "figures"))
    ensure_dirs(results_dir, figures_dir)
    methods = cfg.get("faithfulness_methods", cfg.get("methods", ["hyb_fpde", "bayesian_hyb_fpde"]))
    frames = []
    for seed in cfg.get("seeds", [0]):
        for task_id, payload, split_name in load_tabular_openml_or_local(cfg, seed=int(seed), mode=args.mode):
            dataset_name, X_train, y_train, X_test, y_test, model, feature_names, _ = payload
            _, metrics, _ = evaluate_methods_for_dataset(
                dataset_name=dataset_name,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                model=model,
                feature_names=feature_names,
                methods=methods,
                seed=int(seed),
                task_id=task_id,
                fold=split_name,
                n_explain=int(cfg.get("n_explain", 20)),
                posterior_samples=int(cfg.get("posterior_samples", 100)),
                top_k=int(cfg.get("top_k", 5)),
                lambda_hyb=float(cfg.get("lambda_hyb", 0.5)),
            )
            frames.append(metrics)
            logger.info("faithfulness completed dataset=%s seed=%s", dataset_name, seed)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    write_csv(df, results_dir / "faithfulness_metrics.csv")
    save_line_plot(df, x="explained_order", y="deletion_auc", group="method", path=figures_dir / "deletion_curves_examples.png", title="Deletion AUC examples")
    save_line_plot(df, x="explained_order", y="insertion_auc", group="method", path=figures_dir / "insertion_curves_examples.png", title="Insertion AUC examples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
