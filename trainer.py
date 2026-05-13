"""
Model trainer — XGBoost with walk-forward cross-validation.

Key design decisions (mention all of these in interviews):

1. Walk-forward CV — never train on future data to predict the past.
   Each fold trains on years T-N to T, tests on T+1.

2. SMOTE only on training folds — applying it to the test set would
   inflate performance. Classic mistake that juniors make.

3. Class-weighted XGBoost + SMOTE together — belt and suspenders for
   the 95:5 class imbalance.

4. Evaluate on PR-AUC not accuracy — accuracy is meaningless here.

5. SHAP for explainability — every prediction comes with a reason.

6. MLflow for experiment tracking — reproducible, auditable.
"""

import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost
import shap
import xgboost as xgb
import optuna
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    classification_report,
)
from sklearn.preprocessing import StandardScaler

from database import SessionLocal, FeatureVector

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

FEATURE_COLS = [
    "uncertainty_ratio", "hedging_ratio", "negation_ratio",
    "first_person_ratio", "positive_ratio", "negative_ratio",
    "uncertainty_delta", "hedging_delta", "sentiment_delta",
    "mean_qa_similarity", "evasion_rate", "cfo_evasion_rate", "ceo_evasion_rate",
    "prepared_sentiment", "qa_sentiment", "sentiment_gap",
]

MLFLOW_EXPERIMENT = "earnings-deception-detector"
MODEL_OUTPUT_DIR  = Path("models")


class DeceptionModelTrainer:

    def __init__(self):
        MODEL_OUTPUT_DIR.mkdir(exist_ok=True)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # ---------------------------------------------------------------- #
    # Data loading                                                      #
    # ---------------------------------------------------------------- #

    def load_features(self) -> pd.DataFrame:
        """Load all labelled feature vectors from the DB."""
        db = SessionLocal()
        try:
            rows = (
                db.query(FeatureVector)
                .filter(FeatureVector.label.isnot(None))
                .all()
            )
            data = []
            for r in rows:
                row = {col: getattr(r, col) for col in FEATURE_COLS}
                row.update({
                    "ticker": r.ticker,
                    "fiscal_year": r.fiscal_year,
                    "quarter": r.quarter,
                    "label": r.label,
                })
                data.append(row)

            df = pd.DataFrame(data).dropna(subset=FEATURE_COLS)
            log.info(
                f"Loaded {len(df)} samples | "
                f"positives: {df['label'].sum()} ({df['label'].mean():.1%})"
            )
            return df
        finally:
            db.close()

    # ---------------------------------------------------------------- #
    # Walk-forward cross-validation                                     #
    # ---------------------------------------------------------------- #

    def walk_forward_cv(
        self, df: pd.DataFrame, train_start: int = 2014, test_start: int = 2020
    ) -> dict:
        """
        Walk-forward CV: train on [train_start, year-1], test on [year].
        Iterates from test_start to max year in data.

        This is the correct way to evaluate time-series financial models.
        Standard k-fold leaks future information.
        """
        years = sorted(df["fiscal_year"].unique())
        test_years = [y for y in years if y >= test_start]

        all_metrics = []
        all_preds   = []

        for test_year in test_years:
            train_df = df[
                (df["fiscal_year"] >= train_start) &
                (df["fiscal_year"] <  test_year)
            ]
            test_df = df[df["fiscal_year"] == test_year]

            if len(train_df) < 50 or len(test_df) < 5:
                continue
            if train_df["label"].sum() < 5:
                log.warning(f"Too few positives in train set for year {test_year}")
                continue

            X_train = train_df[FEATURE_COLS].values
            y_train = train_df["label"].values
            X_test  = test_df[FEATURE_COLS].values
            y_test  = test_df["label"].values

            # SMOTE on training fold only
            X_train_res, y_train_res = self._apply_smote(X_train, y_train)

            model = self._build_model(y_train)
            model.fit(
                X_train_res, y_train_res,
                eval_set=[(X_test, y_test)],
                verbose=False,
            )

            y_prob = model.predict_proba(X_test)[:, 1]
            pr_auc = average_precision_score(y_test, y_prob)
            roc_auc = roc_auc_score(y_test, y_prob) if y_test.sum() > 0 else 0

            all_metrics.append({
                "test_year": test_year,
                "pr_auc": round(pr_auc, 4),
                "roc_auc": round(roc_auc, 4),
                "n_test": len(y_test),
                "n_pos": int(y_test.sum()),
            })
            all_preds.extend(zip(y_test, y_prob))

            log.info(f"Year {test_year}: PR-AUC={pr_auc:.3f} ROC-AUC={roc_auc:.3f}")

        metrics_df = pd.DataFrame(all_metrics)
        log.info(f"\nMean PR-AUC: {metrics_df['pr_auc'].mean():.4f}")
        log.info(f"Mean ROC-AUC: {metrics_df['roc_auc'].mean():.4f}")

        return {
            "per_year": metrics_df.to_dict("records"),
            "mean_pr_auc": metrics_df["pr_auc"].mean(),
            "mean_roc_auc": metrics_df["roc_auc"].mean(),
        }

    # ---------------------------------------------------------------- #
    # Final model training + MLflow logging                            #
    # ---------------------------------------------------------------- #

    def train_final_model(self, df: pd.DataFrame) -> xgb.XGBClassifier:
        """
        Train on all available data (use after CV confirms model quality).
        Logs everything to MLflow. Returns model + saves SHAP explainer.
        """
        X = df[FEATURE_COLS].values
        y = df["label"].values

        X_res, y_res = self._apply_smote(X, y)
        model = self._build_model(y)

        with mlflow.start_run(run_name=f"final_{datetime.now():%Y%m%d_%H%M}"):
            mlflow.log_params({
                "model_type": "XGBClassifier",
                "smote": True,
                "n_features": len(FEATURE_COLS),
                "train_samples": len(y_res),
                "positive_rate": round(y.mean(), 4),
            })

            model.fit(X_res, y_res, verbose=False)

            # Full dataset metrics (in-sample — report CV metrics as primary)
            y_prob = model.predict_proba(X)[:, 1]
            pr_auc  = average_precision_score(y, y_prob)
            roc_auc = roc_auc_score(y, y_prob)

            mlflow.log_metrics({"train_pr_auc": pr_auc, "train_roc_auc": roc_auc})
            mlflow.xgboost.log_model(model, "xgboost_model")

            # SHAP explainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            shap.summary_plot(
                shap_values, X,
                feature_names=FEATURE_COLS,
                show=False,
                plot_type="bar",
            )
            import matplotlib.pyplot as plt
            plt.tight_layout()
            plt.savefig("models/shap_summary.png", dpi=150, bbox_inches="tight")
            plt.close()
            mlflow.log_artifact("models/shap_summary.png")

            log.info(f"Final model | PR-AUC: {pr_auc:.4f} | ROC-AUC: {roc_auc:.4f}")

        # Save locally too
        model.save_model("models/deception_model.json")
        return model

    # ---------------------------------------------------------------- #
    # Hyperparameter search with Optuna                                #
    # ---------------------------------------------------------------- #

    def tune_hyperparameters(self, df: pd.DataFrame, n_trials: int = 50) -> dict:
        """
        Optuna hyperparameter search optimised for PR-AUC on the
        most recent 2 years of data as the hold-out.
        """
        holdout_year = df["fiscal_year"].max()
        train_df = df[df["fiscal_year"] < holdout_year]
        test_df  = df[df["fiscal_year"] == holdout_year]

        X_train = train_df[FEATURE_COLS].values
        y_train = train_df["label"].values
        X_test  = test_df[FEATURE_COLS].values
        y_test  = test_df["label"].values

        X_train_res, y_train_res = self._apply_smote(X_train, y_train)

        def objective(trial):
            params = {
                "n_estimators":     trial.suggest_int("n_estimators", 100, 800),
                "max_depth":        trial.suggest_int("max_depth", 3, 10),
                "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
                "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            }
            neg = (y_train == 0).sum()
            pos = (y_train == 1).sum()
            model = xgb.XGBClassifier(
                **params,
                scale_pos_weight=neg / pos,
                eval_metric="aucpr",
                
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X_train_res, y_train_res, verbose=False)
            y_prob = model.predict_proba(X_test)[:, 1]
            return average_precision_score(y_test, y_prob)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        log.info(f"Best PR-AUC: {study.best_value:.4f}")
        log.info(f"Best params: {study.best_params}")
        return study.best_params

    # ---------------------------------------------------------------- #
    # Inference                                                         #
    # ---------------------------------------------------------------- #

    def predict(
        self, feature_dict: dict, model: xgb.XGBClassifier
    ) -> dict:
        """
        Score a single transcript's features.
        Returns risk score (0-100) + SHAP explanation.
        """
        X = np.array([[feature_dict.get(c, 0) for c in FEATURE_COLS]])
        prob = model.predict_proba(X)[0][1]
        risk_score = round(prob * 100, 1)

        # SHAP explanation
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)[0]
        explanation = sorted(
            [
                {"feature": f, "shap_value": round(float(s), 4)}
                for f, s in zip(FEATURE_COLS, shap_vals)
            ],
            key=lambda x: abs(x["shap_value"]),
            reverse=True,
        )

        return {
            "risk_score": risk_score,
            "probability": round(float(prob), 4),
            "risk_level": "HIGH" if risk_score > 65 else "MEDIUM" if risk_score > 40 else "LOW",
            "top_drivers": explanation[:5],
        }

    # ---------------------------------------------------------------- #
    # Private helpers                                                   #
    # ---------------------------------------------------------------- #

    def _apply_smote(self, X, y):
        """Apply SMOTE only if there are enough minority samples."""
        n_minority = y.sum()
        if n_minority < 6:
            log.warning("Too few positives for SMOTE — skipping oversampling")
            return X, y
        k = min(5, n_minority - 1)
        sm = SMOTE(k_neighbors=k, random_state=42)
        return sm.fit_resample(X, y)

    def _build_model(self, y_train) -> xgb.XGBClassifier:
        neg = (y_train == 0).sum()
        pos = max((y_train == 1).sum(), 1)
        return xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=neg / pos,   # handles class imbalance
            eval_metric="aucpr",
        
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1,
        )
