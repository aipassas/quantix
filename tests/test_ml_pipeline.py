"""Tests for ml_pipeline.py — the Momentum Continuation classifier's
feature engineering, no-lookahead labeling, time-based split (no temporal
leakage), training, persistence, and serving.

The network boundary (data_loader.load_price_history_only) is stubbed
throughout with synthetic OHLCV data, so these are deterministic. One test
constructs a dataset with a KNOWN, perfectly learnable relationship
between one feature and the label specifically to prove the training
pipeline can actually learn a real signal when one is present — not just
that it runs without crashing.
"""
import datetime

import numpy as np
import pandas as pd
import pytest

import ml_pipeline as mlp
from ml_pipeline import (
    FEATURE_COLUMNS,
    PredictionResult,
    build_features,
    build_labels,
    build_training_dataset,
    load_history,
    load_model,
    predict_latest,
    save_model,
    time_based_split,
    train_momentum_model,
    training_universe,
)


# --- build_features ------------------------------------------------------------

def test_build_features_adds_every_declared_column(clean_ohlcv):
    result = build_features(clean_ohlcv)
    for col in FEATURE_COLUMNS:
        assert col in result.columns


def test_build_features_warmup_rows_are_nan_not_fabricated(clean_ohlcv):
    result = build_features(clean_ohlcv)
    # SMA_50-derived features need 50 bars — row 0 cannot have a real value.
    assert pd.isna(result["sma20_vs_sma50_pct"].iloc[0])


def test_build_features_only_uses_trailing_data(ohlcv_factory):
    """No lookahead: truncating the input after row t must not change any
    feature value AT OR BEFORE row t (a feature that changed would mean
    something downstream of t leaked into it)."""
    full = ohlcv_factory(n=200)
    truncated = full.iloc[:150]

    full_features = build_features(full)
    truncated_features = build_features(truncated)

    pd.testing.assert_frame_equal(
        full_features.loc[truncated_features.index, list(FEATURE_COLUMNS)],
        truncated_features[list(FEATURE_COLUMNS)],
    )


# --- build_labels ----------------------------------------------------------------

def test_build_labels_matches_hand_computed_direction():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    df = pd.DataFrame({"Close": [100.0, 105.0, 95.0, 110.0, 108.0]}, index=idx)
    labels = build_labels(df, horizon_days=1)
    # day0->day1: 100->105 up (1); day1->day2: 105->95 down (0); day2->day3: 95->110 up (1)
    assert labels.iloc[0] == 1.0
    assert labels.iloc[1] == 0.0
    assert labels.iloc[2] == 1.0


def test_build_labels_final_horizon_rows_are_nan():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    df = pd.DataFrame({"Close": [100.0, 101.0, 102.0, 103.0, 104.0]}, index=idx)
    labels = build_labels(df, horizon_days=2)
    assert pd.isna(labels.iloc[-1])
    assert pd.isna(labels.iloc[-2])
    assert not pd.isna(labels.iloc[-3])


# --- training_universe -----------------------------------------------------------

def test_training_universe_matches_watchlist_config_deduplicated():
    from config import WATCHLIST
    universe = training_universe()
    assert set(universe) == set(WATCHLIST.tech_basket) | set(WATCHLIST.diversified_basket)
    assert len(universe) == len(set(universe))  # no duplicates


# --- build_training_dataset (stubbed network) -------------------------------------

def test_build_training_dataset_skips_bad_ticker_with_reason(monkeypatch, ohlcv_factory):
    good = ohlcv_factory(n=200, seed=1)

    def fake_load(ticker, start, end):
        if ticker == "GOOD":
            return good, []
        return pd.DataFrame(), ["no data"]

    monkeypatch.setattr(mlp, "load_price_history_only", fake_load)
    X, y, meta, skipped = build_training_dataset(("GOOD", "BAD"), datetime.date.today(), 400, 10)

    assert not X.empty
    assert "BAD" in skipped
    assert set(meta["ticker"].unique()) == {"GOOD"}


def test_build_training_dataset_row_count_matches_meta_and_labels(monkeypatch, ohlcv_factory):
    data = ohlcv_factory(n=250, seed=2)
    monkeypatch.setattr(mlp, "load_price_history_only", lambda ticker, start, end: (data, []))
    X, y, meta, skipped = build_training_dataset(("A",), datetime.date.today(), 400, 10)

    assert len(X) == len(y) == len(meta)
    assert not y.isna().any()          # only labeled rows survive
    assert not X.isna().any().any()    # only feature-complete rows survive


def test_build_training_dataset_all_tickers_bad_returns_empty_with_reasons(monkeypatch):
    monkeypatch.setattr(mlp, "load_price_history_only", lambda ticker, start, end: (pd.DataFrame(), ["fetch failed"]))
    X, y, meta, skipped = build_training_dataset(("X", "Y"), datetime.date.today(), 400, 10)
    assert X.empty
    assert set(skipped.keys()) == {"X", "Y"}


# --- time_based_split (no temporal leakage) ---------------------------------------

def test_time_based_split_every_train_date_precedes_every_test_date():
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    meta = pd.DataFrame({"ticker": ["A"] * 100, "date": dates})
    train_idx, test_idx = time_based_split(meta, test_fraction=0.2)

    assert len(train_idx) > 0 and len(test_idx) > 0
    assert meta.loc[train_idx, "date"].max() < meta.loc[test_idx, "date"].min()


def test_time_based_split_holds_out_approximately_the_requested_fraction():
    dates = pd.date_range("2024-01-01", periods=200, freq="D")
    meta = pd.DataFrame({"ticker": ["A"] * 200, "date": dates})
    train_idx, test_idx = time_based_split(meta, test_fraction=0.25)
    assert abs(len(test_idx) / 200 - 0.25) < 0.05


def test_time_based_split_interleaved_tickers_still_no_leakage():
    """Two tickers with OVERLAPPING date ranges — the split must still be
    a single global date cutoff, not accidentally per-ticker."""
    dates_a = pd.date_range("2024-01-01", periods=100, freq="D")
    dates_b = pd.date_range("2024-01-15", periods=100, freq="D")  # starts later, ends later
    meta = pd.concat([
        pd.DataFrame({"ticker": "A", "date": dates_a}),
        pd.DataFrame({"ticker": "B", "date": dates_b}),
    ], ignore_index=True)
    train_idx, test_idx = time_based_split(meta, test_fraction=0.2)
    assert meta.loc[train_idx, "date"].max() < meta.loc[test_idx, "date"].min()


# --- train_momentum_model: refusal paths ---------------------------------------

def test_train_refuses_when_below_minimum_rows(monkeypatch, ohlcv_factory):
    tiny = ohlcv_factory(n=65, seed=3)  # barely enough to warm up features, nowhere near min_training_rows
    monkeypatch.setattr(mlp, "load_price_history_only", lambda ticker, start, end: (tiny, []))
    model, result = train_momentum_model(tickers=("A",))
    assert model is None
    assert result.error is not None
    assert "minimum" in result.error or "below" in result.error


def test_train_refuses_gracefully_when_all_tickers_fail(monkeypatch):
    monkeypatch.setattr(mlp, "load_price_history_only", lambda ticker, start, end: (pd.DataFrame(), ["down"]))
    model, result = train_momentum_model(tickers=("A", "B"))
    assert model is None
    assert result.error is not None


# --- train_momentum_model: the pipeline can learn a REAL signal -----------------

def _make_learnable_dataset(n=600, seed=42):
    """A synthetic price series engineered so that whenever RSI(14) is
    LOW (oversold), price reliably rises over the next 10 days, and
    whenever RSI is HIGH (overbought), it reliably falls — a real,
    learnable relationship the model should be able to pick up on well
    above chance. This is not a claim about real markets; it's a
    correctness check that the pipeline can learn ANY real signal at all
    when one is deliberately present in the data.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n)
    prices = [100.0]
    # Oscillate deterministically between an "oversold, about to rise" and
    # "overbought, about to fall" regime in blocks, with real noise on top.
    regime_len = 20
    for i in range(1, n):
        regime = (i // regime_len) % 2  # 0 = rising regime, 1 = falling regime
        drift = 0.006 if regime == 0 else -0.006
        prices.append(prices[-1] * (1 + drift + rng.normal(0, 0.004)))
    close = np.array(prices)
    volume = rng.integers(1_000_000, 5_000_000, n)
    df = pd.DataFrame({
        "Open": close, "High": close * 1.001, "Low": close * 0.999, "Close": close, "Volume": volume,
    }, index=dates)
    return df


def test_pipeline_learns_a_real_signal_when_one_exists(monkeypatch):
    """The strongest correctness check: feed the full pipeline a dataset
    with a genuine, strong, learnable relationship and confirm the fitted
    model's out-of-sample accuracy clears the naive majority-class
    baseline by a wide margin. If this fails, the pipeline itself
    (feature building, labeling, splitting, or fitting) is broken —
    not just "the model has no edge," which is an expected, fine
    outcome on REAL market data but should never happen here."""
    dataset = _make_learnable_dataset()
    monkeypatch.setattr(mlp, "load_price_history_only", lambda ticker, start, end: (dataset, []))

    model, result = train_momentum_model(tickers=("SYN",), end_date=datetime.date(2022, 6, 1))

    assert result.error is None
    assert model is not None
    assert result.test_accuracy > result.majority_class_baseline_accuracy + 0.1
    assert result.test_roc_auc is not None and result.test_roc_auc > 0.7


# --- persistence -----------------------------------------------------------------

def test_save_and_load_model_round_trip(tmp_path, monkeypatch, ohlcv_factory):
    data = ohlcv_factory(n=600, seed=5)
    monkeypatch.setattr(mlp, "load_price_history_only", lambda ticker, start, end: (data, []))
    model, result = train_momentum_model(tickers=("A", "B"))
    assert model is not None  # sanity: fixture must actually train

    model_path = tmp_path / "model.joblib"
    history_path = tmp_path / "history.json"
    save_model(model, result, model_path, history_path)

    loaded_model = load_model(model_path)
    assert loaded_model is not None

    history = load_history(history_path)
    assert len(history) == 1
    assert history[0]["test_accuracy"] == pytest.approx(result.test_accuracy)


def test_load_model_missing_file_returns_none(tmp_path):
    assert load_model(tmp_path / "nope.joblib") is None


def test_load_history_missing_file_returns_empty_list(tmp_path):
    assert load_history(tmp_path / "nope.json") == []


def test_load_history_corrupt_file_degrades_to_empty_not_raise(tmp_path):
    path = tmp_path / "history.json"
    path.write_text("{not valid json")
    assert load_history(path) == []


def test_save_model_appends_not_overwrites_history(tmp_path, monkeypatch, ohlcv_factory):
    data = ohlcv_factory(n=600, seed=6)
    monkeypatch.setattr(mlp, "load_price_history_only", lambda ticker, start, end: (data, []))
    model, result = train_momentum_model(tickers=("A",))
    model_path = tmp_path / "model.joblib"
    history_path = tmp_path / "history.json"

    save_model(model, result, model_path, history_path)
    save_model(model, result, model_path, history_path)

    assert len(load_history(history_path)) == 2


# --- predict_latest ----------------------------------------------------------------

def test_predict_latest_no_model_status():
    result = predict_latest("AAPL", pd.DataFrame(), None)
    assert result.status == "no_model"
    assert result.probability_up is None


def test_predict_latest_insufficient_data_status(clean_ohlcv):
    tiny = clean_ohlcv.iloc[:10]
    fake_model = object()  # any non-None sentinel; predict_proba is never reached
    result = predict_latest("AAPL", tiny, fake_model)
    assert result.status == "insufficient_data"


def test_predict_latest_ok_status_returns_valid_probability(monkeypatch, ohlcv_factory):
    data = ohlcv_factory(n=600, seed=7)
    monkeypatch.setattr(mlp, "load_price_history_only", lambda ticker, start, end: (data, []))
    model, result = train_momentum_model(tickers=("A",))
    assert model is not None

    prediction = predict_latest("AAPL", data, model)
    assert prediction.status == "ok"
    assert 0.0 <= prediction.probability_up <= 1.0
    assert set(prediction.feature_values.keys()) == set(FEATURE_COLUMNS)
    assert prediction.as_of_date == str(data.index[-1].date())
