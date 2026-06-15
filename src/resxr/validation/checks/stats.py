"""
Basic statistics for ResXR pipeline.
Includes functions to compute mean, median, standard deviation, min, max
for the data being processed in the ResXR pipeline.

Runs on each stream separately. For every numeric column (excluding timestamp
columns) this check computes and logs:
  count, NaN count, NaN %, mean, median, std, min, max, p5, p25, p75, p95.

No quality flags are emitted — this is a purely informational check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...core.config import ValidationConfig
from ...core.constants import GLOBAL_CLOCK_COLUMN
from ...core.logger import get_logger
from ...core.session import QualityFlag, Session, TrackingStream
from ..registry import register_check

logger = get_logger(__name__)

# Timestamp-like columns that should be excluded from numeric stats
_TIMESTAMP_COLS = {"timestamp", GLOBAL_CLOCK_COLUMN, "Eyes_Time"}


def compute_column_stats(series: pd.Series) -> dict[str, float]:
    """
    Compute descriptive statistics for a single numeric column.

    Parameters
    ----------
    series : pd.Series
        A numeric column (NaNs are ignored for aggregate statistics).

    Returns
    -------
    dict
        Keys: count, nan_count, nan_pct, mean, median, std, min,
              p5, p25, p75, p95, max.
    """
    n_total = len(series)
    nan_count = int(series.isna().sum())
    valid = series.dropna()
    n_valid = len(valid)

    if n_valid == 0:
        return {
            "count": n_total,
            "nan_count": nan_count,
            "nan_pct": 100.0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "p5": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
        }

    arr = valid.to_numpy(dtype=float)
    return {
        "count": n_total,
        "nan_count": nan_count,
        "nan_pct": round(nan_count / n_total * 100, 2) if n_total > 0 else 0.0,
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1) if n_valid > 1 else 0.0),
        "min": float(np.min(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def compute_stream_stats(
    stream: TrackingStream,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute per-column descriptive statistics for an entire stream.

    Parameters
    ----------
    stream : TrackingStream
        The tracking stream to summarise.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - Summary DataFrame: Stream-level summary (e.g., total rows, columns).
        - Detailed DataFrame: Per-column stats (e.g., mean, std, etc.).
    """
    df = stream.data
    numeric_cols = [
        col for col in df.select_dtypes(include=[np.number]).columns if col not in _TIMESTAMP_COLS
    ]

    # Compute detailed stats for each column
    detailed_stats = pd.DataFrame({col: compute_column_stats(df[col]) for col in numeric_cols}).T

    # Compute stream-level summary
    summary_stats = pd.DataFrame(
        {
            "row_count": [len(df)],
            "column_count": [len(numeric_cols)],
            "nan_pct": [
                round(df[numeric_cols].isna().sum().sum() / (len(df) * len(numeric_cols)) * 100, 2)
                if len(df) > 0 and len(numeric_cols) > 0
                else 0.0
            ],
        }
    )

    return summary_stats, detailed_stats


class StatsSummaryCheck:
    """
    Compute and save descriptive statistics for every numeric column in a stream.

    For each column the following are computed:
    count, NaN count, NaN %, mean, median, std, min, p5, p25, p75, p95, max.

    No quality flags are emitted — this check is purely informational.
    """

    name = "stats_summary"
    description = "Per-column descriptive statistics for each tracking stream"
    required_streams = None

    def __call__(
        self,
        stream: TrackingStream,
        session: Session,
        config: ValidationConfig,
    ) -> list[QualityFlag]:
        """Run statistics summary on the stream and save results."""
        summary_stats, detailed_stats = compute_stream_stats(stream)

        # Save stats to the stream object for later use in the report
        stream.stats_summary = summary_stats
        stream.stats_detailed = detailed_stats

        return []  # Purely informational — no flags raised


# Register the check
stats_summary_check = StatsSummaryCheck()
register_check(stats_summary_check)
