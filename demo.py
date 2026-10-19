"""CPU forecasting demo with a held-out future and a standalone HTML plot."""

import argparse
import csv
from pathlib import Path
from time import perf_counter

import numpy as np
import plotly.graph_objects as go
import torch
from timestral import TimestralForecaster


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, help="CSV with equally spaced observations")
    parser.add_argument("--column", default="value", help="Numeric CSV column")
    parser.add_argument("--history", type=int, default=504, help="Synthetic history length")
    parser.add_argument("--horizon", type=int, default=48, help="Final observations to hide")
    parser.add_argument("--period", type=int, default=24, help="Baseline's seasonal period")
    parser.add_argument("--noise", type=float, default=4, help="Synthetic noise strength")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("forecast.html"))
    args = parser.parse_args()
    if not 1 <= args.horizon <= 1024:
        parser.error("--horizon must be between 1 and 1024")
    if args.period < 1 or args.threads < 1 or args.noise < 0:
        parser.error("--period and --threads must be positive; --noise must be nonnegative")
    if not 32 <= args.history <= 32768:
        parser.error("--history must be between 32 and 32768")

    if args.csv:
        with args.csv.open(newline="") as source:
            reader = csv.DictReader(source)
            if args.column not in (reader.fieldnames or []):
                parser.error(f"CSV has no {args.column!r} column")
            try:
                values = np.array([float(row[args.column]) for row in reader], dtype=np.float32)
            except (ValueError, TypeError) as error:
                parser.error(f"CSV column must contain numbers: {error}")
        label = args.csv.name
    else:
        t = np.arange(args.history + args.horizon)
        rng = np.random.default_rng(args.seed)
        values = (
            120 + 0.08 * t + 35 * np.sin(2 * np.pi * (t - 8) / 24)
            + 8 * np.cos(2 * np.pi * t / 168) + rng.normal(0, args.noise, len(t))
        ).astype(np.float32)
        label = "synthetic hourly server traffic"

    if len(values) < args.horizon + max(32, args.period):
        parser.error("Need at least horizon + max(32, period) observations")
    if not np.isfinite(values).all():
        parser.error("This small demo expects finite values without missing rows")
    context, actual = values[:-args.horizon], values[-args.horizon:]
    if len(context) > 32768:
        parser.error("Use at most 32768 historical observations")
    future_x = np.arange(len(context), len(values))
    baseline = np.resize(context[-args.period:], args.horizon)

    torch.set_num_threads(args.threads)
    print("Loading mistralai/Timestral-5M on CPU (downloads about 21 MB on first run)...", flush=True)
    model = TimestralForecaster.from_pretrained("mistralai/Timestral-5M", device="cpu")
    start = perf_counter()
    with torch.inference_mode():
        prediction = model.forecast(context, prediction_length=args.horizon)
    elapsed = perf_counter() - start
    median = np.asarray(prediction.median).reshape(-1)
    lower = np.asarray(prediction.quantile(0.1)).reshape(-1)
    upper = np.asarray(prediction.quantile(0.9)).reshape(-1)
    assert median.shape == lower.shape == upper.shape == actual.shape
    assert np.isfinite(median).all() and np.isfinite(lower).all() and np.isfinite(upper).all()
    assert np.all(lower <= median) and np.all(median <= upper)
    mae = float(np.mean(np.abs(actual - median)))
    baseline_mae = float(np.mean(np.abs(actual - baseline)))
    coverage = float(np.mean((actual >= lower) & (actual <= upper)))

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=future_x, y=lower, mode="lines", line_width=0, showlegend=False, hoverinfo="skip"))
    figure.add_trace(go.Scatter(x=future_x, y=upper, mode="lines", line_width=0,
                               fill="tonexty", fillcolor="rgba(52,211,153,0.2)", name="10–90% quantile range"))
    history_x = np.arange(len(context))
    figure.add_trace(go.Scatter(x=history_x, y=context, name="History", line_color="#94a3b8"))
    figure.add_trace(go.Scatter(x=future_x, y=actual, name="Actual held-out values", line_color="#fbbf24"))
    figure.add_trace(go.Scatter(x=future_x, y=baseline, name=f"Repeat last {args.period} steps",
                               line=dict(color="#a78bfa", dash="dot")))
    figure.add_trace(go.Scatter(x=future_x, y=median, name="Timestral median", line_color="#34d399"))
    figure.add_vline(x=len(context) - 0.5, line_dash="dash", line_color="#64748b")
    figure.update_layout(
        title=dict(text=f"Timestral-5M · {label}", subtitle=dict(text=(
            f"MAE {mae:.2f} · baseline MAE {baseline_mae:.2f} · "
            f"range coverage {coverage:.0%} · inference {elapsed * 1000:.0f} ms"
        ))),
        template="plotly_dark", height=620, hovermode="x unified",
        xaxis_title="Observation step", yaxis_title="Value", dragmode="pan",
        xaxis=dict(range=[max(0, len(context) - 168), len(values) - 1], rangeslider=dict(visible=True)),
        legend=dict(orientation="h", y=-0.35), margin=dict(t=100, b=120),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(args.output, include_plotlyjs=True, config={"scrollZoom": True})
    with args.output.with_suffix(".csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["step", "actual", "p10", "median", "p90", "seasonal_baseline"])
        writer.writerows(zip(future_x, actual, lower, median, upper, baseline))
    if not args.csv:
        with args.output.with_name("example.csv").open("w", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(["step", "value"])
            writer.writerows(enumerate(values))
    print(f"Forecast MAE: {mae:.3f}; seasonal baseline MAE: {baseline_mae:.3f}")
    print(f"Inference: {elapsed * 1000:.1f} ms; held-out range coverage: {coverage:.1%}")
    print(f"Open {args.output.resolve()}")
    print(f"Forecast data: {args.output.with_suffix('.csv').resolve()}")


if __name__ == "__main__":
    main()
