"""Machine Learning Pipeline — a Momentum Continuation classifier.

Every other analysis section in Quantix is rule-based and backward-
looking (a Z-Score crosses a fixed threshold, RSI enters a fixed zone).
This module adds the one genuinely predictive component: given a
ticker's current technical/risk state, estimate the probability its
price will be higher `ML_PIPELINE.label_horizon_days` trading days from
now than it is today.

DESIGN, MATCHING THE ORIGINATING TASK'S OWN WORDING:
  - "Baseline model" — a single Logistic Regression on standardized
    features, not an ensemble or deep model. Financial time series are
    noisy and a Logistic Regression's coefficients stay inspectable
    (you can literally ask "does the model think RSI matters, and in
    which direction"), which a black-box model would hide.
  - "Reuses existing fundamental/technical/risk data" — every feature
    below is computed via technical_indicators.py / risk_analytics.py,
    the exact same functions the rest of the app already uses. No new
    indicator math exists in this module.
  - "Track model performance over time and support retraining" — the
    fitted model and every training run's metrics are persisted to local
    files (joblib + JSON), the same local-file-store pattern
    realtime_alerts.py already established for this app's one other
    piece of cross-restart state. Retraining is a rerun of the same
    pipeline, appending a new history entry rather than silently
    replacing the old one.

WHY THIS IS HARD, STATED PLAINLY RATHER THAN HIDDEN: predicting stock
price DIRECTION is close to the textbook definition of a hard problem —
liquid-market prices already reflect public information, so a baseline
model trained only on public technical features should not be expected
to show a large, reliable edge. This module's whole reporting posture is
built around that: every prediction is shown next to the SAME baseline
this app compares every other strategy against (a naive "always predict
the majority class" accuracy, mirroring "vs Buy & Hold" everywhere else
in this app), and the UI is written to never imply the output is
investment advice or a guaranteed edge — it's an estimated probability
from a specific, disclosed, backtested model, nothing more.

NO LOOKAHEAD, BY CONSTRUCTION: every feature at row t is computed from
data available UP TO AND INCLUDING day t (trailing windows only); the
label at row t looks FORWARD to day t+horizon and is therefore NaN (and
dropped) for the most recent `horizon_days` rows of any ticker's history
— there is no way for a label to leak into its own feature row. The
train/test split is by DATE across the whole combined dataset, never a
random shuffle, which would let a training row's neighbors from the same
ticker leak adjacent-day information into the test set.
"""
import datetime
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import CHART_DEFAULTS, ML_PIPELINE, RISK, WATCHLIST
from data_loader import load_price_history_only
from local_store import atomic_write_text, shared_path
from logging_setup import get_logger, log_event, log_exception
from price_processing import process_price_data
from risk_analytics import compute_rolling_volatility
from technical_indicators import compute_macd, compute_rsi, compute_sma_lines

logger = get_logger("ml_pipeline")

# The exact, ORDERED feature set both training and serving use. Order
# matters: a fitted sklearn Pipeline has no column names of its own, only
# positions — training and prediction must build this vector identically,
# which is why both paths funnel through the single build_features()
# function below rather than each assembling their own column list.
FEATURE_COLUMNS: Tuple[str, ...] = (
    "rsi_14",
    "macd_histogram",
    "price_vs_sma20_pct",
    "sma20_vs_sma50_pct",
    "return_10d_pct",
    "return_20d_pct",
    "volatility_20d",
    "volume_ratio_20d",
)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Every feature column, trailing-window only — no feature at row t
    reads anything dated after t. `df` must already be process_price_data()
    output (a clean OHLCV frame with a Close/Volume column)."""
    out = df.copy()
    out = compute_sma_lines(out, [20, 50])
    out = compute_macd(out)
    out["rsi_14"] = compute_rsi(out, 14)
    out["macd_histogram"] = out["MACD_Line"] - out["MACD_Signal"]
    out["price_vs_sma20_pct"] = (out["Close"] - out["SMA_20"]) / out["SMA_20"] * 100
    out["sma20_vs_sma50_pct"] = (out["SMA_20"] - out["SMA_50"]) / out["SMA_50"] * 100
    out["return_10d_pct"] = out["Close"].pct_change(10) * 100
    out["return_20d_pct"] = out["Close"].pct_change(20) * 100
    out["volatility_20d"] = compute_rolling_volatility(out, window=20, trading_days_per_year=RISK.trading_days_per_year)
    volume_avg_20d = out["Volume"].rolling(window=20, min_periods=20).mean()
    out["volume_ratio_20d"] = out["Volume"] / volume_avg_20d
    return out


def build_labels(df: pd.DataFrame, horizon_days: int) -> pd.Series:
    """1 if Close is higher `horizon_days` trading days FORWARD of each
    row, 0 if not, NaN for the final `horizon_days` rows (no future data
    exists yet for them — dropped by the caller, never fabricated)."""
    forward_close = df["Close"].shift(-horizon_days)
    label = (forward_close > df["Close"]).astype(float)
    label[forward_close.isna()] = np.nan
    return label


def training_universe() -> Tuple[str, ...]:
    """The same fixed, deduplicated ticker universe risk_alerts.py's
    watchlist_tickers() already scans — reused rather than a second
    hand-picked list, so this module doesn't invent its own notion of
    "the market" separate from what the rest of the app already treats as
    its scoring universe."""
    return tuple(dict.fromkeys(WATCHLIST.tech_basket + WATCHLIST.diversified_basket))


@dataclass
class TrainingRow:
    ticker: str
    date: pd.Timestamp


def build_training_dataset(
    tickers: Tuple[str, ...],
    end_date: datetime.date,
    lookback_days: int,
    horizon_days: int,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, Dict[str, str]]:
    """Fetch + feature-engineer + label every ticker in `tickers`
    independently, then concatenate into one combined (X, y, meta) triple.
    `meta` carries the ticker/date each row came from — needed for the
    date-based train/test split and for tracing a prediction back to its
    source. Returns (X, y, meta, skipped) where `skipped` discloses any
    ticker that couldn't contribute rows and why, never silently dropped.
    """
    start_date = end_date - datetime.timedelta(days=lookback_days)
    frames: List[pd.DataFrame] = []
    labels: List[pd.Series] = []
    metas: List[pd.DataFrame] = []
    skipped: Dict[str, str] = {}

    for ticker in tickers:
        price_history, errors = load_price_history_only(ticker, start_date, end_date)
        if errors or price_history.empty:
            skipped[ticker] = "; ".join(errors) or "no price history returned"
            continue

        cleaned = process_price_data(price_history, ticker=ticker).df
        if cleaned.empty or len(cleaned) < 60:
            skipped[ticker] = f"only {len(cleaned)} usable bar(s) after cleaning — not enough to warm up features"
            continue

        featured = build_features(cleaned)
        label = build_labels(featured, horizon_days)

        row_mask = featured[list(FEATURE_COLUMNS)].notna().all(axis=1) & label.notna()
        if not row_mask.any():
            skipped[ticker] = "no rows survived feature warm-up + label horizon"
            continue

        frames.append(featured.loc[row_mask, list(FEATURE_COLUMNS)])
        labels.append(label.loc[row_mask])
        metas.append(pd.DataFrame({"ticker": ticker, "date": featured.index[row_mask]}))

    if not frames:
        return pd.DataFrame(columns=FEATURE_COLUMNS), pd.Series(dtype=float), pd.DataFrame(columns=["ticker", "date"]), skipped

    X = pd.concat(frames, ignore_index=True)
    y = pd.concat(labels, ignore_index=True)
    meta = pd.concat(metas, ignore_index=True)
    return X, y, meta, skipped


def time_based_split(meta: pd.DataFrame, test_fraction: float) -> Tuple[np.ndarray, np.ndarray]:
    """Row indices for train/test, split by a DATE cutoff across the
    WHOLE combined dataset — every training row's date is strictly before
    every test row's date, regardless of which ticker it came from. A
    random shuffle (sklearn's default train_test_split) would let one
    ticker's day-40 row train on information adjacent to another row's
    day-41 test label; a per-ticker split would still let one ticker's
    LATER dates leak into training while another's EARLIER dates sit in
    test. A single global date cutoff avoids both.
    """
    unique_dates = pd.Series(meta["date"].unique()).sort_values().reset_index(drop=True)
    cutoff_idx = max(1, int(len(unique_dates) * (1 - test_fraction)))
    cutoff_idx = min(cutoff_idx, len(unique_dates) - 1)
    cutoff_date = unique_dates.iloc[cutoff_idx]
    train_idx = meta.index[meta["date"] < cutoff_date].to_numpy()
    test_idx = meta.index[meta["date"] >= cutoff_date].to_numpy()
    return train_idx, test_idx


@dataclass
class TrainingResult:
    trained_at: str
    tickers_used: Tuple[str, ...]
    tickers_skipped: Dict[str, str]
    label_horizon_days: int
    train_rows: int
    test_rows: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_accuracy: float
    test_accuracy: float
    test_roc_auc: Optional[float]
    majority_class_baseline_accuracy: float
    feature_columns: Tuple[str, ...] = FEATURE_COLUMNS
    error: Optional[str] = None


def train_momentum_model(
    tickers: Optional[Tuple[str, ...]] = None,
    end_date: Optional[datetime.date] = None,
) -> Tuple[Optional[Pipeline], Optional[TrainingResult]]:
    """The full pipeline: fetch -> feature-engineer -> label -> time-split
    -> fit -> evaluate. Returns (None, TrainingResult-with-error) instead
    of raising for every input-shaped failure (too little data, every
    ticker skipped), matching this app's established (value, reason)
    convention — never a crash, always a disclosed reason.
    """
    tickers = tickers or training_universe()
    end_date = end_date or datetime.date.today()

    X, y, meta, skipped = build_training_dataset(tickers, end_date, ML_PIPELINE.train_lookback_days, ML_PIPELINE.label_horizon_days)

    tickers_used = tuple(sorted(set(meta["ticker"]))) if not meta.empty else ()
    if len(X) < ML_PIPELINE.min_training_rows:
        reason = (
            f"only {len(X)} usable training row(s), below the {ML_PIPELINE.min_training_rows} minimum — "
            + ("; ".join(f"{t}: {r}" for t, r in skipped.items()) if skipped else "not enough tickers produced usable data")
        )
        log_event(logger, logging.WARNING, "ml_pipeline.train_refused", rows=len(X), reason=reason)
        return None, TrainingResult(
            trained_at=datetime.datetime.now().isoformat(timespec="seconds"),
            tickers_used=tickers_used, tickers_skipped=skipped, label_horizon_days=ML_PIPELINE.label_horizon_days,
            train_rows=0, test_rows=0, train_start="", train_end="", test_start="", test_end="",
            train_accuracy=0.0, test_accuracy=0.0, test_roc_auc=None, majority_class_baseline_accuracy=0.0,
            error=reason,
        )

    train_idx, test_idx = time_based_split(meta, ML_PIPELINE.test_fraction)
    if len(train_idx) == 0 or len(test_idx) == 0:
        reason = "date-based split produced an empty train or test set — need a wider date range or more tickers"
        return None, TrainingResult(
            trained_at=datetime.datetime.now().isoformat(timespec="seconds"),
            tickers_used=tickers_used, tickers_skipped=skipped, label_horizon_days=ML_PIPELINE.label_horizon_days,
            train_rows=len(train_idx), test_rows=len(test_idx), train_start="", train_end="", test_start="", test_end="",
            train_accuracy=0.0, test_accuracy=0.0, test_roc_auc=None, majority_class_baseline_accuracy=0.0,
            error=reason,
        )

    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_test, y_test = X.loc[test_idx], y.loc[test_idx]

    try:
        # class_weight="balanced": financial up/down labels are rarely a
        # perfect 50/50 split (a bull-biased universe skews toward more
        # "up" rows) — without this, a model can inflate its accuracy by
        # trivially always predicting the majority class, which is exactly
        # the failure mode majority_class_baseline_accuracy below is
        # computed to catch and disclose if it still happens.
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])
        pipeline.fit(X_train, y_train)
    except Exception as e:
        log_exception(logger, "ml_pipeline.fit_error", section="ml_pipeline")
        reason = f"model fitting failed: {type(e).__name__}: {e}"
        return None, TrainingResult(
            trained_at=datetime.datetime.now().isoformat(timespec="seconds"),
            tickers_used=tickers_used, tickers_skipped=skipped, label_horizon_days=ML_PIPELINE.label_horizon_days,
            train_rows=len(X_train), test_rows=len(X_test), train_start="", train_end="", test_start="", test_end="",
            train_accuracy=0.0, test_accuracy=0.0, test_roc_auc=None, majority_class_baseline_accuracy=0.0,
            error=reason,
        )

    train_pred = pipeline.predict(X_train)
    test_pred = pipeline.predict(X_test)
    test_proba = pipeline.predict_proba(X_test)[:, 1]

    majority_class = y_test.mode().iloc[0] if not y_test.mode().empty else 0.0
    majority_baseline_acc = float((y_test == majority_class).mean())

    try:
        test_auc = float(roc_auc_score(y_test, test_proba)) if y_test.nunique() > 1 else None
    except ValueError:
        test_auc = None

    result = TrainingResult(
        trained_at=datetime.datetime.now().isoformat(timespec="seconds"),
        tickers_used=tickers_used, tickers_skipped=skipped, label_horizon_days=ML_PIPELINE.label_horizon_days,
        train_rows=len(X_train), test_rows=len(X_test),
        train_start=str(meta.loc[train_idx, "date"].min().date()), train_end=str(meta.loc[train_idx, "date"].max().date()),
        test_start=str(meta.loc[test_idx, "date"].min().date()), test_end=str(meta.loc[test_idx, "date"].max().date()),
        train_accuracy=float(accuracy_score(y_train, train_pred)),
        test_accuracy=float(accuracy_score(y_test, test_pred)),
        test_roc_auc=test_auc,
        majority_class_baseline_accuracy=majority_baseline_acc,
    )
    log_event(
        logger, logging.INFO, "ml_pipeline.trained",
        train_rows=result.train_rows, test_rows=result.test_rows,
        test_accuracy=round(result.test_accuracy, 4), baseline=round(result.majority_class_baseline_accuracy, 4),
        tickers=len(tickers_used),
    )
    return pipeline, result


# --- Persistence: local files, same pattern as realtime_alerts.py's store ----

def _model_path() -> Path:
    return shared_path(ML_PIPELINE.model_filename)


def _history_path() -> Path:
    return shared_path(ML_PIPELINE.history_filename)


def save_model(model: Pipeline, result: TrainingResult, model_path: Optional[Path] = None, history_path: Optional[Path] = None) -> None:
    """Persists the fitted model (joblib) and appends this run's metrics
    to the training history (JSON, atomic write) — never overwrites prior
    history, so "track model performance over time" has something to show."""
    model_path = model_path or _model_path()
    history_path = history_path or _history_path()

    joblib.dump(model, model_path)

    history = load_history(history_path)
    history.append(result.__dict__)
    history = history[-ML_PIPELINE.max_history:]

    atomic_write_text(history_path, json.dumps(history, indent=2, default=str))


def load_history(history_path: Optional[Path] = None) -> List[dict]:
    """Never raises: a missing or corrupt history file is treated as an
    empty history rather than crashing the app on load."""
    history_path = history_path or _history_path()
    if not history_path.exists():
        return []
    try:
        return json.loads(history_path.read_text())
    except Exception:
        log_exception(logger, "ml_pipeline.history_corrupt", section="ml_pipeline")
        return []


def load_model(model_path: Optional[Path] = None) -> Optional[Pipeline]:
    """None (never raises) if no model has been trained yet, or the saved
    file is unreadable — the caller is responsible for prompting a first
    training run rather than this function fabricating a model."""
    model_path = model_path or _model_path()
    if not model_path.exists():
        return None
    try:
        return joblib.load(model_path)
    except Exception:
        log_exception(logger, "ml_pipeline.model_load_error", section="ml_pipeline")
        return None


# --- Serving -------------------------------------------------------------

@dataclass
class PredictionResult:
    ticker: str
    status: str  # "ok" | "insufficient_data" | "no_model"
    detail: str = ""
    probability_up: Optional[float] = None
    as_of_date: Optional[str] = None
    feature_values: Dict[str, float] = field(default_factory=dict)


def predict_latest(ticker: str, price_history: pd.DataFrame, model: Optional[Pipeline]) -> PredictionResult:
    """The model's estimated probability `ticker`'s price is higher
    ML_PIPELINE.label_horizon_days trading days from the LAST available
    bar than it is today. Never fabricates a probability when the
    features genuinely can't be computed (too little history) or no
    model has been trained — returns a disclosed status instead.
    """
    if model is None:
        return PredictionResult(ticker=ticker, status="no_model", detail="No trained model available yet — train one first.")

    if price_history.empty or len(price_history) < 60:
        return PredictionResult(ticker=ticker, status="insufficient_data", detail=f"only {len(price_history)} bar(s) available — need at least 60 to warm up every feature")

    cleaned = process_price_data(price_history, ticker=ticker).df
    featured = build_features(cleaned)
    latest = featured.iloc[-1]

    feature_vector = latest[list(FEATURE_COLUMNS)]
    if feature_vector.isna().any():
        missing = [c for c in FEATURE_COLUMNS if pd.isna(latest[c])]
        return PredictionResult(ticker=ticker, status="insufficient_data", detail=f"feature(s) not yet computable: {', '.join(missing)}")

    X = pd.DataFrame([feature_vector.to_dict()], columns=FEATURE_COLUMNS)
    probability_up = float(model.predict_proba(X)[0, 1])

    return PredictionResult(
        ticker=ticker, status="ok", probability_up=probability_up,
        as_of_date=str(featured.index[-1].date()),
        feature_values={c: float(feature_vector[c]) for c in FEATURE_COLUMNS},
    )
