from __future__ import annotations

import argparse
import random
from datetime import datetime
from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "DataSet"
CLEAN_DIR = DATA_DIR / "cleaned"


def _sorted_unique_numeric(series: pd.Series) -> list[int]:
    values = pd.to_numeric(series, errors="coerce").dropna().astype(int).unique().tolist()
    return sorted(values)


def _require_clean_file(path: Path) -> None:
    if not path.exists():
        raise SystemExit(
            f"Missing cleaned file: {path.name}. Run Preprocessing.py first to generate cleaned datasets."
        )


def _critical_value(sample_size: int, alpha: float = 0.05) -> float | None:
    if sample_size < 2:
        return None
    if sample_size >= 30:
        return 1.96
    try:
        from scipy import stats
    except ImportError:
        return 1.96
    return stats.t.ppf(1 - alpha / 2, df=sample_size - 1)


def _half_width(values: list[float]) -> float | None:
    series = pd.Series(values).dropna()
    if len(series) < 2:
        return None
    critical = _critical_value(len(series))
    if critical is None:
        return None
    return critical * series.std(ddof=1) / (len(series) ** 0.5)


def _compute_metrics(results_df: pd.DataFrame, start_seconds: pd.Series | None) -> dict:
    if results_df.empty:
        return {
            "sim_mean": float("nan"),
            "obs_mean": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
            "mape": float("nan"),
            "run_length_hours": 0.0,
            "throughput_per_hr": 0.0,
            "count": 0,
        }

    mean_obs = results_df["observed_duration_mins"].mean()
    mean_sim = results_df["simulated_duration_mins"].mean()
    mae = (results_df["simulated_duration_mins"] - results_df["observed_duration_mins"]).abs().mean()
    rmse = ((results_df["simulated_duration_mins"] - results_df["observed_duration_mins"]) ** 2).mean() ** 0.5

    valid = results_df["observed_duration_mins"].gt(0)
    if valid.any():
        mape = (
            (results_df.loc[valid, "simulated_duration_mins"]
            - results_df.loc[valid, "observed_duration_mins"]).abs()
            / results_df.loc[valid, "observed_duration_mins"]
        ).mean() * 100.0
    else:
        mape = float("nan")

    run_length_hours = 0.0
    if start_seconds is not None:
        start_seconds = start_seconds.reset_index(drop=True)
        if len(start_seconds) == len(results_df):
            end_seconds = start_seconds + results_df["simulated_duration_mins"].to_numpy() * 60.0
            run_length_hours = end_seconds.max() / 3600.0 if len(end_seconds) else 0.0

    throughput_per_hr = len(results_df) / run_length_hours if run_length_hours else 0.0

    return {
        "sim_mean": mean_sim,
        "obs_mean": mean_obs,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "run_length_hours": run_length_hours,
        "throughput_per_hr": throughput_per_hr,
        "count": len(results_df),
    }


def _write_siman_report(
    results_df: pd.DataFrame,
    trips_df: pd.DataFrame,
    seed_base: int,
    replications: int,
    run_length_hours: float,
    rep_metrics: list[dict],
    report_path: Path,
) -> None:
    merged = trips_df[["trip_id", "date", "start_time"]].merge(
        results_df[["trip_id", "simulated_duration_mins", "observed_duration_mins"]],
        on="trip_id",
        how="inner",
    )

    sim = merged["simulated_duration_mins"].dropna()
    obs = merged["observed_duration_mins"].dropna()
    abs_err = (sim - obs).abs()

    rep_df = pd.DataFrame(rep_metrics)
    rep_counts = results_df.groupby("replication").size().rename("count")
    if not rep_df.empty:
        rep_df = rep_df.merge(rep_counts, left_on="replication", right_index=True, how="left")

    required_cols = ["sim_mean", "obs_mean", "mae", "rmse", "mape", "run_length_hours", "count"]
    rep_df = rep_df.dropna(subset=[col for col in required_cols if col in rep_df.columns])

    sim_means = rep_df["sim_mean"].tolist() if "sim_mean" in rep_df else []
    obs_means = rep_df["obs_mean"].tolist() if "obs_mean" in rep_df else []
    mae_means = rep_df["mae"].tolist() if "mae" in rep_df else []
    rmse_means = rep_df["rmse"].tolist() if "rmse" in rep_df else []
    mape_means = rep_df["mape"].tolist() if "mape" in rep_df else []

    half_sim = _half_width(sim_means)
    half_obs = _half_width(obs_means)
    half_mae = _half_width(mae_means)

    def weighted_mean(values: pd.Series, weights: pd.Series) -> float | None:
        if values.empty or weights.empty:
            return None
        if weights.nunique(dropna=True) <= 1:
            return float(values.mean())
        return float((values * weights).sum() / weights.sum())

    sim_mean = weighted_mean(rep_df["sim_mean"], rep_df["count"]) if "sim_mean" in rep_df else None
    obs_mean = weighted_mean(rep_df["obs_mean"], rep_df["count"]) if "obs_mean" in rep_df else None
    mae_mean = weighted_mean(rep_df["mae"], rep_df["count"]) if "mae" in rep_df else None
    rmse_mean = weighted_mean(rep_df["rmse"], rep_df["count"]) if "rmse" in rep_df else None
    mape_mean = weighted_mean(rep_df["mape"], rep_df["count"]) if "mape" in rep_df else None
    if "throughput_per_hr" in rep_df and not rep_df.empty:
        throughput_mean = float(rep_df["throughput_per_hr"].mean())
    else:
        throughput_mean = 0.0

    stats = {
        "sim_mean": sim_mean if sim_mean is not None else float("nan"),
        "sim_min": float(pd.Series(sim_means).min()) if sim_means else float("nan"),
        "sim_max": float(pd.Series(sim_means).max()) if sim_means else float("nan"),
        "sim_std": float(pd.Series(sim_means).std(ddof=1)) if len(sim_means) > 1 else float("nan"),
        "sim_var": float(pd.Series(sim_means).var(ddof=1)) if len(sim_means) > 1 else float("nan"),
        "obs_mean": obs_mean if obs_mean is not None else float("nan"),
        "obs_min": float(pd.Series(obs_means).min()) if obs_means else float("nan"),
        "obs_max": float(pd.Series(obs_means).max()) if obs_means else float("nan"),
        "obs_std": float(pd.Series(obs_means).std(ddof=1)) if len(obs_means) > 1 else float("nan"),
        "obs_var": float(pd.Series(obs_means).var(ddof=1)) if len(obs_means) > 1 else float("nan"),
        "abs_mean": mae_mean if mae_mean is not None else float("nan"),
        "abs_min": float(pd.Series(mae_means).min()) if mae_means else float("nan"),
        "abs_max": float(pd.Series(mae_means).max()) if mae_means else float("nan"),
        "mae": mae_mean if mae_mean is not None else float("nan"),
        "rmse": rmse_mean if rmse_mean is not None else float("nan"),
        "mape": mape_mean if mape_mean is not None else float("nan"),
        "throughput_per_hr": throughput_mean,
        "observations": len(sim),
    }

    def half_width_text(value: float | None) -> str:
        return "(Insuf)" if value is None else f"{value:.4f}"

    def fmt_row(cols, widths, aligns):
        parts = []
        for col, width, align in zip(cols, widths, aligns):
            text = str(col)
            if align == "right":
                parts.append(text.rjust(width))
            else:
                parts.append(text.ljust(width))
        return " ".join(parts)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append("*******************************************************************************")
    lines.append("                             ARENA SIMULATION RESULTS")
    lines.append("                             Project: Route 654 Current System")
    if replications == 1:
        lines.append("                             Replication 1 of 1")
    else:
        lines.append(f"                             Replication Summary ({replications} Replications)")
    lines.append(f"                             Report Generated: {timestamp}")
    lines.append("*******************************************************************************")
    lines.append("")
    lines.append("Simulation Run Parameters")
    lines.append("-------------------------------------------------------------------------------")
    lines.append(f"Replication Length                 : {run_length_hours:.2f} Hours")
    lines.append("Warm-Up Period                     : 0.00 Hours")
    lines.append("Base Time Units                    : Minutes")
    lines.append(f"Number of Replications             : {replications}")
    if replications == 1:
        lines.append(f"Random Seed                        : {seed_base}")
    else:
        lines.append(f"Random Seed Base                   : {seed_base}")
    lines.append("")
    lines.append("*******************************************************************************")
    lines.append("")
    lines.append("ENTITY STATISTICS")
    lines.append("-------------------------------------------------------------------------------")
    lines.append("Identifier                         Average     Half Width    Minimum    Maximum")
    lines.append("-------------------------------------------------------------------------------")
    lines.append(
        fmt_row(
            [
                "BusTrip.TimeInSystem",
                f"{stats['sim_mean']:.4f}",
                half_width_text(half_sim),
                f"{stats['sim_min']:.4f}",
                f"{stats['sim_max']:.4f}",
            ],
            [35, 11, 12, 9, 9],
            ["left", "right", "right", "right", "right"],
        )
    )
    lines.append(
        fmt_row(
            [
                "BusTrip.ObservedDuration",
                f"{stats['obs_mean']:.4f}",
                half_width_text(half_obs),
                f"{stats['obs_min']:.4f}",
                f"{stats['obs_max']:.4f}",
            ],
            [35, 11, 12, 9, 9],
            ["left", "right", "right", "right", "right"],
        )
    )
    lines.append(
        fmt_row(
            [
                "BusTrip.AbsoluteError",
                f"{stats['abs_mean']:.4f}",
                half_width_text(half_mae),
                f"{stats['abs_min']:.4f}",
                f"{stats['abs_max']:.4f}",
            ],
            [35, 11, 12, 9, 9],
            ["left", "right", "right", "right", "right"],
        )
    )
    lines.append("")
    lines.append("*******************************************************************************")
    lines.append("")
    lines.append("QUEUE STATISTICS")
    lines.append("-------------------------------------------------------------------------------")
    lines.append("Identifier                         Average     Maximum     Current")
    lines.append("-------------------------------------------------------------------------------")
    lines.append("StopQueue_All                      N/A         N/A         N/A")
    lines.append("")
    lines.append("*******************************************************************************")
    lines.append("")
    lines.append("RESOURCE STATISTICS")
    lines.append("-------------------------------------------------------------------------------")
    lines.append("Identifier                         Scheduled   Busy        Utilization")
    lines.append("-------------------------------------------------------------------------------")
    lines.append("BusServer                          N/A         N/A         N/A")
    lines.append("")
    lines.append("*******************************************************************************")
    lines.append("")
    lines.append("COUNTER STATISTICS")
    lines.append("-------------------------------------------------------------------------------")
    lines.append("Identifier                         Count")
    lines.append("-------------------------------------------------------------------------------")
    lines.append(f"Trips Arrived                      {stats['observations']}")
    lines.append(f"Trips Completed                    {stats['observations']}")
    lines.append(f"Number Observed                    {stats['observations']}")
    lines.append("")
    lines.append("*******************************************************************************")
    lines.append("")
    lines.append("TALLY STATISTICS")
    lines.append("-------------------------------------------------------------------------------")
    lines.append("Identifier                         Average     Half Width   Observations")
    lines.append("-------------------------------------------------------------------------------")
    lines.append(
        fmt_row(
            [
                "TripDurationError",
                f"{stats['mae']:.4f}",
                half_width_text(half_mae),
                f"{stats['observations']}",
            ],
            [35, 11, 12, 12],
            ["left", "right", "right", "right"],
        )
    )
    lines.append("")
    lines.append("*******************************************************************************")
    lines.append("")
    lines.append("INPUT PARAMETERS")
    lines.append("-------------------------------------------------------------------------------")
    lines.append("Arrival Process                    Observed trip start times")
    lines.append("Run-Time Distribution              Empirical (by direction, segment)")
    lines.append("Dwell-Time Distribution            Empirical (by direction, stop)")
    lines.append("Number of Servers                  N/A (not modeled)")
    lines.append("")
    lines.append("*******************************************************************************")
    lines.append("")
    lines.append("SYSTEM PERFORMANCE SUMMARY")
    lines.append("-------------------------------------------------------------------------------")
    lines.append(f"Average Time in System             {stats['sim_mean']:.4f} Minutes")
    lines.append(f"Observed Mean Time in System       {stats['obs_mean']:.4f} Minutes")
    lines.append(f"MAE                                {stats['mae']:.4f} Minutes")
    lines.append(f"RMSE                               {stats['rmse']:.4f} Minutes")
    lines.append(f"MAPE                               {stats['mape']:.4f} %")
    lines.append(f"Throughput                         {stats['throughput_per_hr']:.4f} Trips/Hour")
    lines.append("Average Queue Length (Lq)          N/A")
    lines.append("Average Waiting Time (Wq)          N/A")
    lines.append("Maximum Queue Length               N/A")
    lines.append("System Stability Status            Not Applicable (No Explicit Queue Model)")
    lines.append("")
    lines.append("*******************************************************************************")
    lines.append("")
    lines.append("STATISTICAL OUTPUTS")
    lines.append("-------------------------------------------------------------------------------")
    lines.append(f"Simulated Duration Std Dev         {stats['sim_std']:.4f} Minutes")
    lines.append(f"Simulated Duration Variance        {stats['sim_var']:.4f} Minutes^2")
    lines.append(f"Observed Duration Std Dev          {stats['obs_std']:.4f} Minutes")
    lines.append(f"Observed Duration Variance         {stats['obs_var']:.4f} Minutes^2")
    if replications == 1:
        lines.append("Confidence Intervals               Insufficient (1 replication)")
    else:
        lines.append("Confidence Intervals               95% CI (t-distribution)")
    lines.append("")
    lines.append("*******************************************************************************")
    lines.append("")
    lines.append("END OF SIMULATION REPORT")
    lines.append("")
    lines.append("*******************************************************************************")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def _write_tables(results_df: pd.DataFrame, rep_metrics: list[dict], output_dir: Path) -> None:
    if not rep_metrics:
        return

    rep_count = len(rep_metrics)
    trips_simulated = len(results_df)

    metrics_df = pd.DataFrame(rep_metrics)
    rep_counts = results_df.groupby("replication").size().rename("count")
    metrics_df = metrics_df.merge(rep_counts, left_on="replication", right_index=True, how="left")

    required_cols = [
        "obs_mean",
        "sim_mean",
        "mae",
        "rmse",
        "mape",
        "run_length_hours",
        "throughput_per_hr",
        "count",
    ]
    metrics_df = metrics_df.dropna(subset=required_cols)
    if metrics_df.empty:
        return

    def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
        if weights.nunique(dropna=True) <= 1:
            return float(values.mean())
        return float((values * weights).sum() / weights.sum())

    mean_obs = weighted_mean(metrics_df["obs_mean"], metrics_df["count"])
    mean_sim = weighted_mean(metrics_df["sim_mean"], metrics_df["count"])
    mean_mae = weighted_mean(metrics_df["mae"], metrics_df["count"])
    mean_rmse = weighted_mean(metrics_df["rmse"], metrics_df["count"])
    mean_mape = weighted_mean(metrics_df["mape"], metrics_df["count"])
    mean_throughput = float(metrics_df["throughput_per_hr"].mean())

    validation_table = pd.DataFrame(
        [
            {"Metric": "Replications", "Value": rep_count, "Units": "count"},
            {"Metric": "Trips simulated", "Value": trips_simulated, "Units": "count"},
            {"Metric": "Observed mean duration", "Value": mean_obs, "Units": "mins"},
            {"Metric": "Simulated mean duration", "Value": mean_sim, "Units": "mins"},
            {"Metric": "Mean Absolute Error", "Value": mean_mae, "Units": "mins"},
            {"Metric": "Root Mean Squared Error", "Value": mean_rmse, "Units": "mins"},
            {"Metric": "Mean Absolute Percentage Error", "Value": mean_mape, "Units": "percent"},
            {"Metric": "Throughput", "Value": mean_throughput, "Units": "trips/hour"},
        ]
    )
    validation_csv = output_dir / "baseline_table_validation.csv"
    validation_txt = output_dir / "baseline_table_validation.txt"
    validation_table.to_csv(validation_csv, index=False)
    validation_txt.write_text(
        validation_table.fillna("N/A").to_string(index=False),
        encoding="utf-8",
    )

    def dist_row(label: str, series: pd.Series) -> dict:
        series = series.dropna()
        iqr = series.quantile(0.75) - series.quantile(0.25)
        return {
            "Distribution": label,
            "Min": series.min(),
            "P25": series.quantile(0.25),
            "Median": series.median(),
            "P75": series.quantile(0.75),
            "Max": series.max(),
            "Mean": series.mean(),
            "StdDev": series.std(ddof=1),
            "IQR": iqr,
            "Skewness": series.skew(),
        }

    distribution_table = pd.DataFrame(
        [
            dist_row("Observed", results_df["observed_duration_mins"]),
            dist_row("Simulated", results_df["simulated_duration_mins"]),
        ]
    )
    distribution_csv = output_dir / "baseline_table_distribution.csv"
    distribution_txt = output_dir / "baseline_table_distribution.txt"
    distribution_table.to_csv(distribution_csv, index=False)
    distribution_txt.write_text(
        distribution_table.fillna("N/A").to_string(index=False),
        encoding="utf-8",
    )

    if rep_metrics:
        replication_table = metrics_df[
            [
                "replication",
                "obs_mean",
                "sim_mean",
                "mae",
                "rmse",
                "mape",
                "throughput_per_hr",
            ]
        ].copy()
        replication_csv = output_dir / "baseline_table_replications.csv"
        replication_txt = output_dir / "baseline_table_replications.txt"
        replication_table.to_csv(replication_csv, index=False)
        replication_txt.write_text(
            replication_table.fillna("N/A").to_string(index=False),
            encoding="utf-8",
        )


def _simulate_once(
    trips_df: pd.DataFrame,
    segments_by_dir: dict,
    stops_by_dir: dict,
    run_by_segment: dict,
    dwell_by_stop: dict,
    run_by_dir: dict,
    dwell_by_dir: dict,
    simpy_module,
    seed: int,
    replication_id: int,
) -> tuple[pd.DataFrame, dict, float]:
    rng = random.Random(seed)

    results: list[dict] = []

    def bus_trip(env: "simpy.Environment", trip_row: pd.Series) -> None:
        start_delay = float(trip_row["start_seconds"]) if pd.notna(trip_row["start_seconds"]) else 0.0
        yield env.timeout(start_delay)
        start_time = env.now

        direction = int(trip_row["direction"])
        segments = segments_by_dir.get(direction, [])
        stops = stops_by_dir.get(direction, [])

        for index, segment in enumerate(segments):
            segment_key = (direction, segment)
            run_times = run_by_segment.get(segment_key) or run_by_dir.get(direction, [])
            # Empirical discrete sampling with replacement for run-time delays.
            run_time = rng.choice(run_times) if run_times else 0.0
            yield env.timeout(float(run_time))

            if index < len(stops):
                stop_key = (direction, stops[index])
                dwell_times = dwell_by_stop.get(stop_key) or dwell_by_dir.get(direction, [])
                # Empirical discrete sampling with replacement for dwell-time delays.
                dwell_time = rng.choice(dwell_times) if dwell_times else 0.0
                yield env.timeout(float(dwell_time))

        sim_duration_mins = (env.now - start_time) / 60.0
        results.append(
            {
                "replication": replication_id,
                "trip_id": trip_row["trip_id"],
                "direction": int(trip_row["direction"]),
                "observed_duration_mins": trip_row["observed_duration_mins"],
                "simulated_duration_mins": sim_duration_mins,
            }
        )

    env = simpy_module.Environment()
    for _, trip_row in trips_df.iterrows():
        env.process(bus_trip(env, trip_row))

    env.run()

    results_df = pd.DataFrame(results)
    metrics = _compute_metrics(results_df, trips_df["start_seconds"])
    metrics["replication"] = replication_id
    run_length_hours = metrics["run_length_hours"]

    return results_df, metrics, run_length_hours


def run_baseline_simulation(
    seed_base: int = 42,
    trip_limit: int | None = None,
    replications: int = 1,
) -> None:
    try:
        import simpy
    except ImportError as exc:
        raise SystemExit("Missing dependency: simpy. Install it with `pip install simpy`.") from exc

    if replications < 1:
        raise SystemExit("Replications must be at least 1.")

    dwell_path = CLEAN_DIR / "bus_dwell_times_654_clean.csv"
    run_path = CLEAN_DIR / "bus_running_times_654_clean.csv"
    trips_path = CLEAN_DIR / "bus_trips_654_clean.csv"

    for path in (dwell_path, run_path, trips_path):
        _require_clean_file(path)

    dwell_df = pd.read_csv(dwell_path)
    run_df = pd.read_csv(run_path)
    trips_df = pd.read_csv(trips_path)

    if trip_limit is not None:
        trips_df = trips_df.head(trip_limit).copy()

    dwell_df["bus_stop"] = pd.to_numeric(dwell_df["bus_stop"], errors="coerce").astype("Int64")
    run_df["segment"] = pd.to_numeric(run_df["segment"], errors="coerce").astype("Int64")

    segments_by_dir = {
        direction: _sorted_unique_numeric(group["segment"])
        for direction, group in run_df.dropna(subset=["segment"]).groupby("direction")
    }
    stops_by_dir = {
        direction: _sorted_unique_numeric(group["bus_stop"])
        for direction, group in dwell_df.dropna(subset=["bus_stop"]).groupby("direction")
    }

    run_by_segment = (
        run_df.dropna(subset=["segment", "run_time_in_seconds"])
        .groupby(["direction", "segment"])["run_time_in_seconds"]
        .apply(list)
        .to_dict()
    )
    dwell_by_stop = (
        dwell_df.dropna(subset=["bus_stop", "dwell_time_in_seconds"])
        .groupby(["direction", "bus_stop"])["dwell_time_in_seconds"]
        .apply(list)
        .to_dict()
    )

    run_by_dir = run_df.groupby("direction")["run_time_in_seconds"].apply(list).to_dict()
    dwell_by_dir = dwell_df.groupby("direction")["dwell_time_in_seconds"].apply(list).to_dict()

    start_dt = pd.to_datetime(trips_df["date"] + " " + trips_df["start_time"], errors="coerce")
    start_origin = start_dt.min()
    trips_df["start_seconds"] = (start_dt - start_origin).dt.total_seconds()
    trips_df["observed_duration_mins"] = pd.to_numeric(
        trips_df["duration_in_mins"], errors="coerce"
    )
    trips_df = trips_df.sort_values("start_seconds")

    results_frames: list[pd.DataFrame] = []
    rep_metrics: list[dict] = []

    for replication_id in range(1, replications + 1):
        seed = seed_base + (replication_id - 1)
        results_df, metrics, run_length_hours = _simulate_once(
            trips_df,
            segments_by_dir,
            stops_by_dir,
            run_by_segment,
            dwell_by_stop,
            run_by_dir,
            dwell_by_dir,
            simpy,
            seed,
            replication_id,
        )
        metrics["run_length_hours"] = run_length_hours
        rep_metrics.append(metrics)
        results_frames.append(results_df)

    results_df = pd.concat(results_frames, ignore_index=True)
    results_path = CLEAN_DIR / "baseline_simulation_results.csv"
    results_df.to_csv(results_path, index=False)

    summary_metrics = _compute_metrics(results_df, None)

    summary = pd.DataFrame(
        [
            {
                "replications": replications,
                "trips_simulated": len(results_df),
                "mean_observed_mins": summary_metrics["obs_mean"],
                "mean_simulated_mins": summary_metrics["sim_mean"],
                "mae_mins": summary_metrics["mae"],
                "rmse_mins": summary_metrics["rmse"],
            }
        ]
    )
    summary_path = CLEAN_DIR / "baseline_simulation_summary.csv"
    summary.to_csv(summary_path, index=False)

    report_path = CLEAN_DIR / "baseline_simulation_siman_report.txt"
    avg_run_length_hours = (
        sum(metric["run_length_hours"] for metric in rep_metrics) / len(rep_metrics)
        if rep_metrics
        else 0.0
    )
    _write_siman_report(
        results_df,
        trips_df,
        seed_base,
        replications,
        avg_run_length_hours,
        rep_metrics,
        report_path,
    )

    _write_tables(results_df, rep_metrics, CLEAN_DIR)

    print("Baseline simulation complete. Results saved to:", results_path)
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the baseline current-system simulation.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for replication runs.",
    )
    parser.add_argument(
        "--trip-limit",
        type=int,
        default=None,
        help="Limit number of trips for quick runs.",
    )
    parser.add_argument(
        "--replications",
        type=int,
        default=1,
        help="Number of replications to run.",
    )
    args = parser.parse_args()

    run_baseline_simulation(
        seed_base=args.seed,
        trip_limit=args.trip_limit,
        replications=args.replications,
    )


if __name__ == "__main__":
    main()
