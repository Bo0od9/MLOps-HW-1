from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import logging

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from catboost import CatBoostClassifier

from .preprocess import preprocess_data

logger = logging.getLogger(__name__)


def make_pred():
    cfg = yaml.safe_load(open("configs/config.yaml"))
    inp = Path(cfg["paths"]["input_dir"])
    out = Path(cfg["paths"]["output_dir"])
    models = Path(cfg["paths"]["models_dir"])

    test_path = inp / "test.csv"
    df_test = pd.read_csv(test_path)
    logger.info(f"Loaded {len(df_test)} lines from {test_path}")

    X = preprocess_data(df_test, cfg)
    logger.info(f"Preprocessing ended, {X.shape[1]} features")

    model = CatBoostClassifier()
    model.load_model(models / "model.cbm")

    proba = model.predict_proba(X)[:, 1]
    id_col = cfg["data"].get("id_col", "index")
    pred_col = cfg["data"].get("target_col", "prediction")
    ids = df_test[id_col] if id_col and id_col in df_test.columns else range(len(df_test))
    threshold = cfg["inference"].get("threshold", 0.5)
    labels = (proba >= float(threshold)).astype(int)
    preds = pd.DataFrame({id_col: ids, pred_col: labels})
    output_file_name = cfg["data"].get("output_file_name", "sample_submission.csv")
    preds.to_csv(out / output_file_name, index=False)

    if cfg["inference"].get("produce_feature_importances", True):
        imps: pd.DataFrame = model.get_feature_importance(prettified=True).head(5)  # type: ignore
        imps.set_index("Feature Id")["Importances"].to_json(out / "feature_importances.json", orient="index", indent=4)

    if cfg["inference"].get("produce_density_plot", True):
        plt.figure()
        pd.Series(proba).plot(kind="kde")
        plt.title("Predicted score density")
        plt.xlabel(cfg["data"]["target_col"])
        plt.tight_layout()
        plt.savefig(out / "scores_density.png")
        plt.close()

    logger.info(f"Saved: {out/output_file_name}, {out/'feature_importances.json'}, {out/'scores_density.png'}")


def main():
    make_pred()


if __name__ == "__main__":
    main()
