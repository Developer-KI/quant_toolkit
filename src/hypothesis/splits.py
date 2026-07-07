"""
splits.py — Temporal train/test splits for strategy validation.

Classes:
    HoldoutSplit        — simple date- or fraction-based single split
    WalkForwardSplits   — rolling or expanding walk-forward folds with embargo

All splits operate on Universe objects, preserving L2 and funding data.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterator

import pandas as pd

from core.universe import Universe


@dataclass
class Split:
    fold: int
    train: Universe
    test: Universe
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def __repr__(self) -> str:
        return (
            f"Split(fold={self.fold}, "
            f"train={self.train_start.date()}→{self.train_end.date()}, "
            f"test={self.test_start.date()}→{self.test_end.date()})"
        )


def _slice_universe(universe: Universe, index: pd.DatetimeIndex) -> Universe:
    """Return a Universe whose bars are restricted to the given index."""
    sub = Universe(symbols=universe.symbols)
    for sym in universe.symbols:
        full = universe.ohlcv(sym)
        common = full.index.intersection(index)
        if common.empty:
            continue
        sub_ohlcv = full.loc[common].copy()
        positions = [full.index.get_loc(ts) for ts in common if ts in full.index]

        full_l2 = universe.l2(sym)
        sub_l2 = [full_l2[j] for j in positions if j < len(full_l2)] if full_l2 else None

        full_funding = universe.funding(sym)
        sub_funding = (
            [full_funding[j] for j in positions if j < len(full_funding)]
            if full_funding
            else None
        )
        sub.add_asset(sym, sub_ohlcv, l2=sub_l2, funding=sub_funding)

    for src_name in universe.list_sources():
        sub.add_data_source(universe._aux_sources[src_name])

    return sub


class HoldoutSplit:
    """
    Simple temporal holdout: split a Universe once into train and test.

    Usage:
        train_u, test_u = HoldoutSplit.by_date(universe, test_start="2022-01-01")
        train_u, test_u = HoldoutSplit.by_fraction(universe, test_frac=0.2)
    """

    @staticmethod
    def by_date(
        universe: Universe,
        test_start: str | pd.Timestamp,
    ) -> tuple[Universe, Universe]:
        ref = universe.ohlcv(universe.symbols[0])
        ts = pd.Timestamp(test_start)
        return (
            _slice_universe(universe, ref.index[ref.index < ts]),
            _slice_universe(universe, ref.index[ref.index >= ts]),
        )

    @staticmethod
    def by_fraction(
        universe: Universe,
        test_frac: float = 0.2,
    ) -> tuple[Universe, Universe]:
        ref = universe.ohlcv(universe.symbols[0])
        split = int(len(ref) * (1 - test_frac))
        return (
            _slice_universe(universe, ref.index[:split]),
            _slice_universe(universe, ref.index[split:]),
        )


class WalkForwardSplits:
    """
    Generate sequential train/test folds for walk-forward analysis.

    Args:
        n_splits:       Number of folds to generate.
        method:         "expanding" (train grows) or "rolling" (fixed-size train window).
        train_size:     Training window in bars. Required for "rolling".
        test_size:      Test window in bars. Defaults to auto-divide.
        embargo_bars:   Bars to drop between train end and test start (avoid lookahead).
        min_train_bars: Minimum training bars before the first fold.

    Usage (expanding):
        for split in WalkForwardSplits(n_splits=5).split(universe):
            bt.run(universe=split.train)
            bt.run(universe=split.test)

    Usage (rolling, 252-bar train):
        for split in WalkForwardSplits(method="rolling", train_size=252).split(universe):
            ...
    """

    def __init__(
        self,
        n_splits: int = 5,
        method: str = "expanding",
        train_size: int | None = None,
        test_size: int | None = None,
        embargo_bars: int = 0,
        min_train_bars: int = 50,
    ):
        if method == "rolling" and train_size is None:
            raise ValueError("WalkForwardSplits with method='rolling' requires train_size")
        self.n_splits = n_splits
        self.method = method
        self.train_size = train_size
        self.test_size = test_size
        self.embargo_bars = embargo_bars
        self.min_train_bars = min_train_bars

    def split(self, universe: Universe) -> Iterator[Split]:
        ref = universe.ohlcv(universe.symbols[0])
        idx = ref.index
        n = len(idx)
        test_sz = self.test_size or max(1, (n - self.min_train_bars) // self.n_splits)

        if self.method == "expanding":
            for fold_num in range(self.n_splits):
                test_start_pos = self.min_train_bars + fold_num * test_sz
                test_end_pos = min(test_start_pos + test_sz, n)
                train_end_pos = test_start_pos - self.embargo_bars

                if train_end_pos < self.min_train_bars or test_start_pos >= n:
                    continue

                train_idx = idx[:train_end_pos]
                test_idx = idx[test_start_pos:test_end_pos]

                if train_idx.empty or test_idx.empty:
                    continue

                yield Split(
                    fold=fold_num,
                    train=_slice_universe(universe, train_idx),
                    test=_slice_universe(universe, test_idx),
                    train_start=train_idx[0],
                    train_end=train_idx[-1],
                    test_start=test_idx[0],
                    test_end=test_idx[-1],
                )

        elif self.method == "rolling":
            train_sz = self.train_size
            step = max(1, (n - train_sz) // self.n_splits)

            for fold_num in range(self.n_splits):
                train_start_pos = fold_num * step
                train_end_pos = train_start_pos + train_sz
                test_start_pos = train_end_pos + self.embargo_bars
                test_end_pos = min(test_start_pos + test_sz, n)

                if train_end_pos > n or test_start_pos >= n:
                    break

                train_idx = idx[train_start_pos:train_end_pos]
                test_idx = idx[test_start_pos:test_end_pos]

                if train_idx.empty or test_idx.empty:
                    continue

                yield Split(
                    fold=fold_num,
                    train=_slice_universe(universe, train_idx),
                    test=_slice_universe(universe, test_idx),
                    train_start=train_idx[0],
                    train_end=train_idx[-1],
                    test_start=test_idx[0],
                    test_end=test_idx[-1],
                )

        else:
            raise ValueError(f"Unknown method: {self.method!r}. Use 'expanding' or 'rolling'.")


@dataclass
class TTVSplit:
    """
    Three-way temporal split: train / test / validate.

    Train   — strategy design and IS exploration (do NOT run hypothesis tests here).
    Test    — parameter optimisation (param sweeps); results may inform design.
    Validate— single blind evaluation; all hypothesis tests and performance claims go here.
    """

    train: Universe
    test: Universe
    validate: Universe
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    validate_start: pd.Timestamp
    validate_end: pd.Timestamp

    def __repr__(self) -> str:
        return (
            f"TTVSplit("
            f"train={self.train_start.date()}→{self.train_end.date()}, "
            f"test={self.test_start.date()}→{self.test_end.date()}, "
            f"validate={self.validate_start.date()}→{self.validate_end.date()})"
        )


class TrainTestValidateSplit:
    """
    Build a three-way temporal split from a Universe.

    Usage (fractions):
        ttv = TrainTestValidateSplit.by_fractions(universe, train_frac=0.6, test_frac=0.2)

    Usage (dates):
        ttv = TrainTestValidateSplit.by_dates(
            universe, test_start="2015-01-01", validate_start="2019-01-01"
        )

    embargo_bars: bars dropped from the *start* of each new segment to prevent
    slow-indicator leakage (e.g. a 200-bar EMA still encoding train data at bar 1
    of the test window).  The default of 0 is safe when indicators are recomputed
    fresh per segment; set to your longest indicator period otherwise.
    """

    @staticmethod
    def by_fractions(
        universe: Universe,
        train_frac: float = 0.60,
        test_frac: float = 0.20,
        embargo_bars: int = 0,
    ) -> TTVSplit:
        if train_frac <= 0 or test_frac <= 0 or train_frac + test_frac >= 1.0:
            raise ValueError(
                f"train_frac ({train_frac}) and test_frac ({test_frac}) must both be "
                "positive and sum to less than 1.0."
            )

        ref = universe.ohlcv(universe.symbols[0])
        idx = ref.index
        n = len(idx)

        train_end_pos = int(n * train_frac)
        test_start_pos = train_end_pos + embargo_bars
        test_end_pos = int(n * (train_frac + test_frac))
        validate_start_pos = test_end_pos + embargo_bars

        if test_start_pos >= test_end_pos:
            raise ValueError("Test segment is empty after applying embargo_bars.")
        if validate_start_pos >= n:
            raise ValueError("Validate segment is empty after applying embargo_bars.")

        train_idx = idx[:train_end_pos]
        test_idx = idx[test_start_pos:test_end_pos]
        validate_idx = idx[validate_start_pos:]

        return TTVSplit(
            train=_slice_universe(universe, train_idx),
            test=_slice_universe(universe, test_idx),
            validate=_slice_universe(universe, validate_idx),
            train_start=train_idx[0],
            train_end=train_idx[-1],
            test_start=test_idx[0],
            test_end=test_idx[-1],
            validate_start=validate_idx[0],
            validate_end=validate_idx[-1],
        )

    @staticmethod
    def by_dates(
        universe: Universe,
        test_start: str | pd.Timestamp,
        validate_start: str | pd.Timestamp,
        embargo_bars: int = 0,
    ) -> TTVSplit:
        ts_test = pd.Timestamp(test_start)
        ts_val = pd.Timestamp(validate_start)
        if ts_test >= ts_val:
            raise ValueError("test_start must be before validate_start.")

        ref = universe.ohlcv(universe.symbols[0])
        idx = ref.index

        # Locate boundary positions by date
        test_pos = idx.searchsorted(ts_test)
        val_pos = idx.searchsorted(ts_val)

        train_idx = idx[:test_pos]
        test_idx = idx[test_pos + embargo_bars : val_pos]
        validate_idx = idx[val_pos + embargo_bars :]

        if train_idx.empty:
            raise ValueError("Train segment is empty — move test_start forward.")
        if test_idx.empty:
            raise ValueError(
                "Test segment is empty after applying embargo_bars — "
                "widen the test window or reduce embargo_bars."
            )
        if validate_idx.empty:
            raise ValueError(
                "Validate segment is empty after applying embargo_bars — "
                "move validate_start earlier or reduce embargo_bars."
            )

        return TTVSplit(
            train=_slice_universe(universe, train_idx),
            test=_slice_universe(universe, test_idx),
            validate=_slice_universe(universe, validate_idx),
            train_start=train_idx[0],
            train_end=train_idx[-1],
            test_start=test_idx[0],
            test_end=test_idx[-1],
            validate_start=validate_idx[0],
            validate_end=validate_idx[-1],
        )
