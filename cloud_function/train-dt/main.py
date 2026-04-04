# Decision Tree: train on all data < today (local TZ); hold out today
# HTTP entrypoint: train_dt_http

import os
import io
import json
import logging
import traceback
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from google.cloud import storage
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

# ---- ENV ----
PROJECT_ID = os.getenv("PROJECT_ID", "")
GCS_BUCKET = os.getenv("GCS_BUCKET", "")
DATA_KEY = os.getenv("DATA_KEY", "structured/datasets/listings_master.csv")
OUTPUT_PREFIX = os.getenv("OUTPUT_PREFIX", "preds")  # e.g., "structured/preds"
TIMEZONE = os.getenv("TIMEZONE", "America/New_York")  # split by local day
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")


def _read_csv_from_gcs(client: storage.Client, bucket: str, key: str) -> pd.DataFrame:
    b = client.bucket(bucket)
    blob = b.blob(key)
    if not blob.exists():
        raise FileNotFoundError(f"gs://{bucket}/{key} not found")
    return pd.read_csv(io.BytesIO(blob.download_as_bytes()))


def _write_csv_to_gcs(client: storage.Client, bucket: str, key: str, df: pd.DataFrame) -> None:
    b = client.bucket(bucket)
    blob = b.blob(key)
    blob.upload_from_string(df.to_csv(index=False), content_type="text/csv")


def _upload_file_to_gcs(
    client: storage.Client, bucket: str, key: str, local_path: str, content_type: str
) -> None:
    b = client.bucket(bucket)
    blob = b.blob(key)
    blob.upload_from_filename(local_path, content_type=content_type)


def _clean_numeric(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace(r"[^\d.]+", "", regex=True).str.strip()
    return pd.to_numeric(s, errors="coerce")


def _clean_cat(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip().str.lower()
    s = s.replace({
        "": np.nan,
        "nan": np.nan,
        "none": np.nan,
        "null": np.nan,
    })
    return s


def run_once(dry_run: bool = False, max_depth: int = 12, min_samples_leaf: int = 10):
    client = storage.Client(project=PROJECT_ID)
    df = _read_csv_from_gcs(client, GCS_BUCKET, DATA_KEY)

    required = {"scraped_at", "price", "make", "model", "year", "mileage"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # --- Parse timestamps and choose local-day split ---
    dt = pd.to_datetime(df["scraped_at"], errors="coerce", utc=True)
    df["scraped_at_dt_utc"] = dt
    try:
        df["scraped_at_local"] = df["scraped_at_dt_utc"].dt.tz_convert(TIMEZONE)
    except Exception:
        df["scraped_at_local"] = df["scraped_at_dt_utc"]
    df["date_local"] = df["scraped_at_local"].dt.date

    # --- Clean numerics BEFORE counting/dropping ---
    orig_rows = len(df)
    df["price_num"] = _clean_numeric(df["price"])
    df["year_num"] = _clean_numeric(df["year"])
    df["mileage_num"] = _clean_numeric(df["mileage"])

    for c in [
        "make",
        "model",
        "color",
        "condition",
        "transmission",
        "body_type",
        "fuel_type",
        "drive_type",
        "title_status",
    ]:
        if c in df.columns:
            df[c] = _clean_cat(df[c])

    valid_price_rows = int(df["price_num"].notna().sum())
    logging.info("Rows total=%d | with valid numeric price=%d", orig_rows, valid_price_rows)

    counts = df["date_local"].value_counts().sort_index()
    logging.info(
        "Recent date counts (local): %s",
        json.dumps({str(k): int(v) for k, v in counts.tail(8).items()}),
    )

    unique_dates = sorted(d for d in df["date_local"].dropna().unique())
    if len(unique_dates) < 2:
        return {
            "status": "noop",
            "reason": "need at least two distinct dates",
            "dates": [str(d) for d in unique_dates],
        }

    today_local = unique_dates[-1]
    train_df = df[df["date_local"] < today_local].copy()
    holdout_df = df[df["date_local"] == today_local].copy()

    train_df = train_df[train_df["price_num"].notna()]
    dropped_for_target = int((df["date_local"] < today_local).sum()) - int(len(train_df))
    logging.info(
        "Train rows after target clean: %d (dropped_for_target=%d)",
        len(train_df),
        dropped_for_target,
    )
    logging.info("Holdout rows today (%s): %d", today_local, len(holdout_df))

    if len(train_df) < 40:
        return {"status": "noop", "reason": "too few training rows", "train_rows": int(len(train_df))}

    # --- Model: expanded feature set -> price_num ---
    target = "price_num"

    candidate_cat_cols = [
        "make",
        "model",
        "color",
        "condition",
        "transmission",
        "body_type",
        "fuel_type",
        "drive_type",
        "title_status",
    ]

    candidate_num_cols = [
        "mileage_num",
        "vehicle_age",
        "clean_title_flag",
    ]

    cat_cols = [c for c in candidate_cat_cols if c in train_df.columns]
    num_cols = [c for c in candidate_num_cols if c in train_df.columns]
    feats = cat_cols + num_cols

    logging.info("Using categorical features: %s", cat_cols)
    logging.info("Using numeric features: %s", num_cols)

    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ]
    )

    X_train = train_df[feats]
    y_train = train_df[target]

    base_model = DecisionTreeRegressor(random_state=42)
    pipe = Pipeline([("pre", pre), ("model", base_model)])

    param_grid = {
        "model__max_depth": [6, 10, 14, 18, None],
        "model__min_samples_leaf": [1, 2, 5, 10],
        "model__min_samples_split": [2, 5, 10],
        "model__criterion": ["squared_error", "absolute_error"],
    }

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=3,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)

    pipe = grid.best_estimator_
    best_params = grid.best_params_
    best_cv_mae = float(-grid.best_score_)

    logging.info("Best params from grid search: %s", best_params)
    logging.info("Best CV MAE: %.4f", best_cv_mae)

    # ---- Predict/evaluate on today's holdout ----
    mae_today = None
    rmse_today = None
    mape_today = None
    bias_today = None
    preds_df = pd.DataFrame()
    perm_df = pd.DataFrame()
    pdp_temp_files = []

    if not holdout_df.empty:
        X_h = holdout_df[feats]
        y_hat = pipe.predict(X_h)

        base_cols = [
            "post_id",
            "scraped_at",
            "make",
            "model",
            "year",
            "mileage",
            "price",
            "color",
            "condition",
            "transmission",
            "body_type",
            "fuel_type",
            "drive_type",
            "title_status",
            "clean_title_flag",
            "vehicle_age",
            "miles_per_year",
            "price_per_10k_miles",
        ]
        available_cols = [c for c in base_cols if c in holdout_df.columns]
        preds_df = holdout_df[available_cols].copy()
        preds_df["actual_price"] = holdout_df["price_num"]
        preds_df["pred_price"] = np.round(y_hat, 2)

        if holdout_df["price_num"].notna().any():
            y_true = holdout_df["price_num"]
            mask = y_true.notna()

            if mask.any():
                y_true_valid = y_true[mask]
                X_h_valid = X_h.loc[mask]
                y_hat_valid = y_hat[mask]

                mae_today = float(mean_absolute_error(y_true_valid, y_hat_valid))
                rmse_today = float(np.sqrt(mean_squared_error(y_true_valid, y_hat_valid)))
                bias_today = float(np.mean(y_hat_valid - y_true_valid))

                nonzero_mask = y_true_valid != 0
                if nonzero_mask.any():
                    mape_today = float(
                        np.mean(
                            np.abs(
                                (y_true_valid[nonzero_mask] - y_hat_valid[nonzero_mask])
                                / y_true_valid[nonzero_mask]
                            )
                        ) * 100
                    )

                # Permutation importance on original feature columns
                perm = permutation_importance(
                    pipe,
                    X_h_valid,
                    y_true_valid,
                    n_repeats=5,
                    random_state=42,
                    scoring="neg_mean_absolute_error",
                )

                perm_df = pd.DataFrame(
                    {
                        "feature": feats,
                        "importance_mean": perm.importances_mean,
                        "importance_std": perm.importances_std,
                    }
                ).sort_values("importance_mean", ascending=False)

                # PDPs: numeric-only, top 3 by permutation importance
                top_numeric_features = [
                    feat for feat in perm_df["feature"].tolist()
                    if feat in num_cols
                ][:3]

                for feat in top_numeric_features:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    PartialDependenceDisplay.from_estimator(
                        pipe,
                        X_h_valid,
                        [feat],
                        ax=ax,
                    )
                    ax.set_title(f"PDP: {feat}")

                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        fig.savefig(tmp.name, bbox_inches="tight")
                        pdp_temp_files.append((feat, tmp.name))

                    plt.close(fig)

    # --- Output path: HOURLY folder structure ---
    now_utc = pd.Timestamp.utcnow().tz_convert("UTC")
    out_dir = f"{OUTPUT_PREFIX}/{now_utc.strftime('%Y%m%d%H')}"
    preds_key = f"{out_dir}/preds.csv"
    perm_key = f"{out_dir}/perm_importance.csv"

    if not dry_run and len(preds_df) > 0:
        _write_csv_to_gcs(client, GCS_BUCKET, preds_key, preds_df)
        logging.info("Wrote predictions to gs://%s/%s (%d rows)", GCS_BUCKET, preds_key, len(preds_df))
    else:
        logging.info("Dry run or no holdout rows; skip preds write. Would write to gs://%s/%s", GCS_BUCKET, preds_key)

    if not dry_run and len(perm_df) > 0:
        _write_csv_to_gcs(client, GCS_BUCKET, perm_key, perm_df)
        logging.info("Wrote permutation importance to gs://%s/%s (%d rows)", GCS_BUCKET, perm_key, len(perm_df))
    else:
        logging.info("Dry run or no permutation importance; skip perm write. Would write to gs://%s/%s", GCS_BUCKET, perm_key)

    written_pdp_keys = []
    if not dry_run and len(pdp_temp_files) > 0:
        for feat, local_path in pdp_temp_files:
            safe_feat = feat.replace("/", "_")
            pdp_key = f"{out_dir}/pdp_{safe_feat}.png"
            _upload_file_to_gcs(client, GCS_BUCKET, pdp_key, local_path, "image/png")
            written_pdp_keys.append(pdp_key)
            logging.info("Wrote PDP to gs://%s/%s", GCS_BUCKET, pdp_key)
    else:
        logging.info("Dry run or no PDPs generated; skip PDP writes.")

    return {
        "status": "ok",
        "today_local": str(today_local),
        "train_rows": int(len(train_df)),
        "holdout_rows": int(len(holdout_df)),
        "valid_price_rows": valid_price_rows,
        "mae_today": mae_today,
        "rmse_today": rmse_today,
        "mape_today": mape_today,
        "bias_today": bias_today,
        "best_params": best_params,
        "best_cv_mae": best_cv_mae,
        "preds_key": preds_key,
        "perm_importance_key": perm_key,
        "pdp_keys": written_pdp_keys,
        "dry_run": dry_run,
        "timezone": TIMEZONE,
    }


def train_dt_http(request):
    try:
        body = request.get_json(silent=True) or {}
        result = run_once(
            dry_run=bool(body.get("dry_run", False)),
            max_depth=int(body.get("max_depth", 12)),
            min_samples_leaf=int(body.get("min_samples_leaf", 10)),
        )
        code = 200 if result.get("status") in {"ok", "noop"} else 500
        return (json.dumps(result), code, {"Content-Type": "application/json"})
    except Exception as e:
        logging.error("Error: %s", e)
        logging.error("Trace:\n%s", traceback.format_exc())
        return (json.dumps({"status": "error", "error": str(e)}), 500, {"Content-Type": "application/json"})
