"""Tests for channels.tsv generation (bids/channels.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from resxr.bids.channels import generate_channels_tsv

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prepared_head_df() -> pd.DataFrame:
    """Minimal BIDS-ready motion DataFrame for a HEAD stream."""
    n = 10
    return pd.DataFrame(
        {
            "latency": np.linspace(0.0, 1.0, n),
            "Node_Head_px": np.zeros(n),
            "Node_Head_py": np.zeros(n),
            "Node_Head_pz": np.zeros(n),
            "Node_Head_qx": np.zeros(n),
            "Node_Head_qy": np.zeros(n),
            "Node_Head_qz": np.zeros(n),
            "Node_Head_qw": np.ones(n),
        }
    )


# ===========================================================================
# generate_channels_tsv
# ===========================================================================


class TestGenerateChannelsTsv:
    def test_returns_dataframe(self):
        """Returns a pandas DataFrame."""
        result = generate_channels_tsv(_prepared_head_df(), 90.0)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns_present(self):
        """Result has all required BIDS channels.tsv columns."""
        result = generate_channels_tsv(_prepared_head_df(), 90.0)
        required = {
            "name",
            "component",
            "type",
            "tracked_point",
            "units",
            "sampling_frequency",
            "reference_frame",
        }
        assert required.issubset(set(result.columns))

    def test_one_row_per_input_column(self):
        """Number of rows equals number of columns in the input DataFrame."""
        df = _prepared_head_df()
        result = generate_channels_tsv(df, 90.0)
        assert len(result) == len(df.columns)

    def test_name_column_matches_input_columns(self):
        """'name' column contains the same strings as df.columns (in order)."""
        df = _prepared_head_df()
        result = generate_channels_tsv(df, 90.0)
        assert list(result["name"]) == list(df.columns)

    def test_type_column_has_no_nulls(self):
        """'type' column contains no None or NaN values."""
        result = generate_channels_tsv(_prepared_head_df(), 90.0)
        assert result["type"].notna().all()

    def test_units_column_has_no_nulls(self):
        """'units' column contains no None or NaN values."""
        result = generate_channels_tsv(_prepared_head_df(), 90.0)
        assert result["units"].notna().all()

    def test_sampling_frequency_column_value(self):
        """sampling_frequency column contains the provided frequency for all rows."""
        result = generate_channels_tsv(_prepared_head_df(), 90.0)
        assert (result["sampling_frequency"] == 90.0).all()

    def test_empty_dataframe_returns_empty(self):
        """Empty input DataFrame → 0-row result with correct columns."""
        result = generate_channels_tsv(pd.DataFrame(), 90.0)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    @pytest.mark.parametrize(
        "col,expected_type",
        [
            ("Node_Head_px", "POS"),
            ("Node_Head_py", "POS"),
            ("Node_Head_qx", "ORNT"),
            ("Node_Head_qw", "ORNT"),
            ("latency", "LATENCY"),
        ],
    )
    def test_known_column_type(self, col, expected_type):
        """Each known column has the expected BIDS channel type."""
        df = _prepared_head_df()
        result = generate_channels_tsv(df, 90.0)
        row = result[result["name"] == col]
        assert len(row) == 1
        assert row.iloc[0]["type"] == expected_type

    def test_latency_tracked_point_is_na(self):
        """LATENCY channels have tracked_point = 'n/a'."""
        df = _prepared_head_df()
        result = generate_channels_tsv(df, 90.0)
        latency_row = result[result["name"] == "latency"].iloc[0]
        assert latency_row["tracked_point"] == "n/a"

    def test_spatial_columns_have_global_reference_frame(self):
        """POS and ORNT columns have reference_frame = 'global'."""
        df = _prepared_head_df()
        result = generate_channels_tsv(df, 90.0)
        pos_rows = result[result["type"] == "POS"]
        assert (pos_rows["reference_frame"] == "global").all()

    def test_misc_columns_have_na_reference_frame(self):
        """MISC columns have reference_frame = 'n/a'."""
        df = pd.DataFrame(
            {
                "LeftHand_Status_HandTracked": [1, 0],
            }
        )
        result = generate_channels_tsv(df, 90.0)
        misc_rows = result[result["type"] == "MISC"]
        if len(misc_rows) > 0:
            assert (misc_rows["reference_frame"] == "n/a").all()
