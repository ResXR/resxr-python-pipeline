"""
HTML report generator for ResXR.

Produces a dashboard-style quality report.

Time display:

    Quality flags internally store times in per-system ``timestamp``
    space (which may be an alternate per-system clock, not the global
    Unity clock).  For the report, all flag times are converted to the
    global ``timeSinceStartup`` clock via ``np.interp`` and then
    expressed relative to the global recording onset (first non-zero
    ``timeSinceStartup`` across all streams).  This gives a single
    shared timeline starting from 0 so events across different tracking
    systems are directly comparable.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader

from ..core.config import PipelineConfig, ReportConfig
from ..core.logger import get_logger
from ..core.session import Session
from ..utils import find_recording_onset

logger = get_logger(__name__)


class ReportGenerator:
    """Generate the dashboard-style quality report."""

    def __init__(self, config: ReportConfig):
        self.config = config
        template_dir = Path(__file__).parent / "templates"
        self._env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=True,
        )

    def _stream_stats(self, session: Session) -> list[dict]:
        """
        Collect stream-level and detailed stats for the report.

        Parameters
        ----------
        session : Session
            The session object containing streams and stats.

        Returns
        -------
        list[dict]
            A list of dictionaries, one per stream, containing:
            - Stream-level summary stats (row count, column count, NaN %).
            - Detailed column-level stats (count, mean, std, etc.).
        """
        stats = []
        for stream in session.streams.values():
            if stream.stats_summary is None or stream.stats_detailed is None:
                logger.warning(f"Stream {stream.system.value} is missing stats. Skipping.")
                continue

            # Calculate NaN percentage for the stream (Q2: guard for nan_pct column)
            nan_pct = round(
                stream.stats_summary["nan_pct"].iloc[0]
                if not stream.stats_summary.empty and "nan_pct" in stream.stats_summary.columns
                else 0.0,
                2,
            )

            stats.append(
                {
                    "name": stream.system.value,  # Use the correct stream name
                    "rows": stream.stats_summary["row_count"].iloc[0],
                    "channels": stream.channel_count,
                    "freq_expected": stream.sampling_frequency,
                    "freq_effective": stream.sampling_frequency_effective,
                    "nan_pct": nan_pct,
                    "warnings": stream.warning_count,
                    "detailed": stream.stats_detailed.reset_index().to_dict(orient="records"),
                }
            )

        return stats

    def generate(
        self,
        session: Session,
        output_path: Path | None = None,
        config: PipelineConfig | None = None,
    ) -> Path:
        """
        Render the HTML quality report for a session.

        Parameters
        ----------
        session : Session
            The processed session containing streams and quality flags.
        output_path : Path | None
            Destination for the HTML file. Falls back to
            ``<output_dir>/<session_id>_report.html`` or the current directory.
        config : PipelineConfig | None
            Pipeline config used to populate device metadata in the report.

        Returns
        -------
        Path
            The path to the written HTML report.
        """
        if output_path is None:
            if self.config.output_dir:
                output_path = self.config.output_dir / f"{session.session_id}_report.html"
            else:
                output_path = Path(f"{session.session_id}_report.html")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        context = {
            "session_id": session.session_id,
            "subject_id": session.subject_id,
            "session_label": session.session_label,
            "utc_start": (
                session.metadata.utc_start.strftime("%Y-%m-%d %H:%M:%S UTC")
                if session.metadata.utc_start
                else "N/A"
            ),
            "device_utc_offset": session.metadata.device_utc_offset or "N/A",
            "manufacturer": config.device.manufacturer if config else "N/A",
            "model_name": config.device.model_name if config else "N/A",
            "platform": session.metadata.platform,
            "unity_version": session.metadata.unity_version,
            "ovrplugin_version": session.metadata.ovrplugin_version or "N/A",
            "build_id": session.metadata.build_id or "N/A",
            "sampling_mode": session.metadata.sampling_mode,
            "fixed_delta_time": session.metadata.fixed_delta_time,
            "schema_rev": session.metadata.schema_rev or "N/A",
            "enabled_systems": [
                name
                for name, enabled in [
                    ("Face", session.metadata.face_enabled),
                    ("Body", session.metadata.body_enabled),
                    ("Hands", session.metadata.hands_enabled),
                    ("Eyes", session.metadata.eyes_enabled),
                    ("Controllers", session.metadata.controllers_enabled),
                ]
                if enabled
            ],
            # Summary stats
            "total_duration": session.total_duration_seconds,
            "total_warnings": session.total_warning_count,
            # Per-stream info
            "streams": self._stream_stats(session),
            # Quality flags (times relative to global recording onset)
            "flags": (flags_rel := self._flags_relative_to_onset(session)),
            # Timeline plot
            "timeline_html": self._build_timeline(
                flags_rel,
                session.total_duration_seconds,
                session.raw_events_data,
            ),
            # Footer
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "resxr_version": __import__("resxr").__version__,
        }
        template = self._env.get_template("report.html")
        rendered = template.render(**context)

        output_path.write_text(rendered, encoding="utf-8")
        logger.info("Report generated: %s", output_path)
        return output_path

    @staticmethod
    def _flags_relative_to_onset(session: Session) -> list[dict]:
        """
        Convert quality flag times to global time relative to recording onset.
        Quality flags internally store times in per-system ``timestamp``
        space (which may be an alternate per-system clock such as
        ``Node_HandLeft_Time``).  To present a single shared timeline in
        the report:

        1. **Global onset** — the earliest ``find_recording_onset`` across
           all streams' ``timeSinceStartup`` columns (the Unity global
           clock).  If a stream is missing ``timeSinceStartup``, a warning
           is logged (no fallback; ``timeSinceStartup`` is the single
           source of truth for cross-system time).

        2. **Per-flag conversion** — for each flag, ``np.interp`` maps
           the per-system start/end times to the global
           ``timeSinceStartup`` axis, then subtracts the global onset.
           This makes all flag times comparable across systems and
           starting from 0.

        Returns a list of dicts consumed by the Jinja2 HTML template.
        """
        # Find global onset from timeSinceStartup (the Unity global clock)
        onsets = []
        for stream in session.streams.values():
            if stream.data.empty:
                continue
            if "timeSinceStartup" not in stream.data.columns:
                logger.warning(
                    f"{stream.system.value}: missing timeSinceStartup column, "
                    "cannot compute global onset for this stream"
                )
                continue
            onset = find_recording_onset(stream.data["timeSinceStartup"].values)
            if onset is not None:
                onsets.append(onset)

        if not onsets:
            logger.warning("No global onset found from timeSinceStartup across any stream")
            global_onset = 0.0
        else:
            global_onset = min(onsets)

        flags = []
        for flag in session.all_flags:
            stream = session.streams.get(flag.system)
            start_global = flag.start_time
            end_global = flag.end_time

            # Convert per-system time → global time when timeSinceStartup exists
            if (
                stream is not None
                and "timeSinceStartup" in stream.data.columns
                and "timestamp" in stream.data.columns
            ):
                ts = stream.data["timestamp"].values
                gs = stream.data["timeSinceStartup"].values
                # Per-system clocks may not be monotonic; np.interp
                # requires sorted xp, so sort both arrays together.
                sort_idx = np.argsort(ts)
                ts_sorted = ts[sort_idx]
                gs_sorted = gs[sort_idx]
                start_global = float(np.interp(flag.start_time, ts_sorted, gs_sorted))
                end_global = float(np.interp(flag.end_time, ts_sorted, gs_sorted))

            flags.append(
                {
                    "id": f"{flag.check_name}_{flag.group_name or 'all'}_{flag.start_time}",
                    "check": flag.check_name,
                    "stream": flag.system.value,
                    "sub_check": flag.group_name or "all",
                    "severity": flag.severity,
                    "start": start_global - global_onset,
                    "end": end_global - global_onset,
                    "duration": end_global - start_global,
                    "group": flag.group_name or "",
                    "message": flag.message,
                }
            )
        return flags

    # Palette for distinguishing check types in the timeline
    _CHECK_COLORS = [
        "#e17055",  # coral
        "#0984e3",  # blue
        "#00b894",  # green
        "#6c5ce7",  # purple
        "#fdcb6e",  # yellow
        "#d63031",  # red
        "#00cec9",  # teal
        "#e84393",  # pink
    ]

    @classmethod
    def _build_timeline(
        cls,
        flags: list[dict],
        total_duration: float,
        events_df: pd.DataFrame | None = None,
    ) -> str:
        """Build a Plotly timeline: thin event row (top) + flag bars (bottom)."""
        from plotly.subplots import make_subplots

        if not flags:
            return ""

        has_events = events_df is not None and not events_df.empty

        # Assign a color per unique check name
        check_names = list(dict.fromkeys(f["check"] for f in flags))
        color_map = {
            name: cls._CHECK_COLORS[i % len(cls._CHECK_COLORS)]
            for i, name in enumerate(check_names)
        }
        stream_names = list(dict.fromkeys(f["stream"] for f in flags))

        # Create subplots: thin events row + main flags row
        fig = make_subplots(
            rows=2,
            cols=1,
            row_heights=[0.15, 0.85] if has_events else [0.001, 0.999],
            shared_xaxes=True,
            vertical_spacing=0.03,
        )

        # === Row 2: quality flag bars ===

        # Invisible anchor traces to stabilise y-axis when toggling
        for name in stream_names:
            fig.add_trace(
                go.Bar(
                    y=[name],
                    x=[0],
                    orientation="h",
                    marker_color="rgba(0,0,0,0)",
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=2,
                col=1,
            )

        # Group flags by (check, sub_check) for individual toggle control
        subcheck_legend_shown: dict[tuple[str, str], bool] = {}
        for f in flags:
            check_name = f["check"]
            sub_check = f["sub_check"]
            key = (check_name, sub_check)

            hover_label = check_name
            if sub_check and sub_check != "all":
                hover_label += f" ({sub_check})"

            show_legend = key not in subcheck_legend_shown
            subcheck_legend_shown[key] = True

            # Legend name shows both check and sub_check for clarity
            legend_name = hover_label

            fig.add_trace(
                go.Bar(
                    y=[f["stream"]],
                    x=[f["duration"]],
                    base=[f["start"]],
                    orientation="h",
                    name=legend_name,
                    legendgroup=f"{check_name}:{sub_check}",  # Unique group per (check, sub_check) as string
                    showlegend=show_legend,
                    hovertemplate=(
                        f"<b>{html.escape(hover_label)}</b><br>"
                        f"Stream: {html.escape(f['stream'])}<br>"
                        f"Time: {f['start']:.1f}s – {f['end']:.1f}s "
                        f"({f['duration']:.1f}s)<br>"
                        f"{html.escape(f['message'])}"
                        "<extra></extra>"
                    ),
                    marker_color=color_map[check_name],
                    opacity=0.7 if sub_check != "all" else 1.0,
                ),
                row=2,
                col=1,
            )

        # === Row 1: events ===
        shapes: list[dict] = []

        if has_events:
            event_name_col = "trial_type" if "trial_type" in events_df.columns else "name"
            instant_events: list[tuple[float, str]] = []
            duration_events: list[tuple[float, float, str]] = []

            for _, row in events_df.iterrows():
                onset = float(row["onset"])
                dur = float(row.get("duration", 0))
                name = str(row.get(event_name_col, ""))
                if dur > 0:
                    duration_events.append((onset, dur, name))
                else:
                    instant_events.append((onset, name))

            # Duration events as bars in the event row
            dur_legend_shown = False
            for onset, dur, name in duration_events:
                fig.add_trace(
                    go.Bar(
                        y=["Events"],
                        x=[dur],
                        base=[onset],
                        orientation="h",
                        name="Event (duration)",
                        legendgroup="event_dur",
                        showlegend=not dur_legend_shown,
                        hovertemplate=(
                            f"<b>{html.escape(name)}</b><br>"
                            f"Onset: {onset:.1f}s, Duration: {dur:.1f}s"
                            "<extra></extra>"
                        ),
                        marker={
                            "color": "rgba(45,52,54,0.4)",
                            "line": {"color": "rgba(45,52,54,0.8)", "width": 1.5},
                        },
                    ),
                    row=1,
                    col=1,
                )
                dur_legend_shown = True

                # Vertical line spanning full chart height
                shapes.append(
                    {
                        "type": "line",
                        "xref": "x",
                        "yref": "paper",
                        "x0": onset,
                        "x1": onset,
                        "y0": 0,
                        "y1": 1,
                        "line": {"color": "rgba(45,52,54,0.35)", "width": 1.5, "dash": "dash"},
                        "layer": "below",
                    }
                )

            # Instant events as diamond markers
            if instant_events:
                onsets = [e[0] for e in instant_events]
                fig.add_trace(
                    go.Scatter(
                        x=onsets,
                        y=["Events"] * len(onsets),
                        mode="markers",
                        name="Event (instant)",
                        legendgroup="event_inst",
                        showlegend=True,
                        marker={
                            "symbol": "diamond",
                            "size": 12,
                            "color": "rgba(45,52,54,0.85)",
                            "line": {"color": "#2d3436", "width": 1.5},
                        },
                        hovertemplate=[
                            f"<b>{html.escape(name)}</b><br>Time: {onset:.1f}s<extra></extra>"
                            for onset, name in instant_events
                        ],
                    ),
                    row=1,
                    col=1,
                )

                for onset, _ in instant_events:
                    shapes.append(
                        {
                            "type": "line",
                            "xref": "x",
                            "yref": "paper",
                            "x0": onset,
                            "x1": onset,
                            "y0": 0,
                            "y1": 1,
                            "line": {"color": "rgba(45,52,54,0.35)", "width": 1.5, "dash": "dash"},
                            "layer": "below",
                        }
                    )

        # === Layout ===
        fig.update_layout(
            barmode="overlay",
            height=140 + 60 * len(stream_names),
            margin={"l": 80, "r": 20, "t": 40, "b": 40},
            plot_bgcolor="#f0f3f8",
            paper_bgcolor="#f0f3f8",
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "left",
                "x": 0,
            },
            shapes=shapes,
        )

        # Bottom x-axis (time label, autoscale)
        fig.update_xaxes(autorange=True, title_text="Time (s)", row=2, col=1)
        # Top x-axis (hidden ticks, shared range)
        fig.update_xaxes(autorange=True, showticklabels=False, row=1, col=1)

        # Y-axis: events row — bold label, distinct background
        if has_events:
            fig.update_yaxes(
                categoryorder="array",
                categoryarray=["Events"],
                tickvals=["Events"],
                ticktext=["<b>Events</b>"],
                tickfont={"size": 13, "color": "#2d3436"},
                row=1,
                col=1,
            )
            fig.update_xaxes(showgrid=False, row=1, col=1)
            # Different background for events subplot
            fig.add_shape(
                type="rect",
                xref="paper",
                yref="paper",
                x0=0,
                x1=1,
                y0=0,
                y1=1,
                fillcolor="rgba(223,230,233,0.5)",
                line_width=0,
                layer="below",
                row=1,
                col=1,
            )
        else:
            fig.update_yaxes(visible=False, row=1, col=1)

        # Y-axis: flags row (stable order)
        fig.update_yaxes(
            categoryorder="array", categoryarray=list(reversed(stream_names)), row=2, col=1
        )

        return fig.to_html(full_html=False, include_plotlyjs="cdn")
