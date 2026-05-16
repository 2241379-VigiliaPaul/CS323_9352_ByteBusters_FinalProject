from __future__ import annotations

from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "DataSet"
OUTPUT_DIR = DATA_DIR / "cleaned"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _coerce_numeric(df: pd.DataFrame, columns: list[str]) -> None:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].dropna().apply(lambda x: float(x).is_integer()).all():
                df[col] = df[col].astype("Int64")


def _normalize_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d")


def _normalize_time(series: pd.Series) -> pd.Series:
    
    return pd.to_datetime(series, errors="coerce").dt.strftime("%H:%M:%S")


def _combine_date_time(date_series: pd.Series, time_series: pd.Series) -> pd.Series:
    combined = date_series.astype(str).str.strip() + " " + time_series.astype(str).str.strip()
    return pd.to_datetime(combined, errors="coerce")


def _adjust_overnight(start_dt: pd.Series, end_dt: pd.Series) -> pd.Series:
    mask = start_dt.notna() & end_dt.notna() & (end_dt < start_dt)
    end_dt = end_dt.where(~mask, end_dt + pd.Timedelta(days=1))
    return end_dt


def _drop_invalid_rows(df: pd.DataFrame, required_cols: list[str]) -> tuple[pd.DataFrame, int]:
    before = len(df)
    df = df.dropna(subset=[c for c in required_cols if c in df.columns])
    return df, before - len(df)


def _write_clean_csv(df: pd.DataFrame, filename: str) -> Path:
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False)
    return path


def clean_bus_dwell_times() -> dict:
    path = DATA_DIR / "bus_dwell_times_654.csv"
    df = pd.read_csv(path)
    original_rows = len(df)

    df["date"] = _normalize_date(df["date"])
    df["arrival_time"] = _normalize_time(df["arrival_time"])
    df["departure_time"] = _normalize_time(df["departure_time"])

    _coerce_numeric(df, ["trip_id", "deviceid", "direction", "dwell_time_in_seconds"])

    arrival_dt = _combine_date_time(df["date"], df["arrival_time"])
    departure_dt = _combine_date_time(df["date"], df["departure_time"])
    departure_dt = _adjust_overnight(arrival_dt, departure_dt)

    computed_dwell = (departure_dt - arrival_dt).dt.total_seconds()
    df["dwell_time_in_seconds"] = df["dwell_time_in_seconds"].where(
        df["dwell_time_in_seconds"].notna() & (df["dwell_time_in_seconds"] >= 0),
        computed_dwell,
    )

    df = df.drop_duplicates()
    df, dropped_invalid = _drop_invalid_rows(
        df,
        ["trip_id", "deviceid", "direction", "bus_stop", "date", "arrival_time", "departure_time"],
    )
    df = df[df["dwell_time_in_seconds"].notna() & (df["dwell_time_in_seconds"] >= 0)]

    _write_clean_csv(df, "bus_dwell_times_654_clean.csv")

    return {
        "file": path.name,
        "rows_in": original_rows,
        "rows_out": len(df),
        "rows_dropped_invalid": dropped_invalid,
        "rows_dropped_duplicates": original_rows - len(df) - dropped_invalid,
    }


def clean_bus_running_times() -> dict:
    path = DATA_DIR / "bus_running_times_654.csv"
    df = pd.read_csv(path)
    original_rows = len(df)

    df["date"] = _normalize_date(df["date"])
    df["start_time"] = _normalize_time(df["start_time"])
    df["end_time"] = _normalize_time(df["end_time"])

    _coerce_numeric(df, ["trip_id", "deviceid", "direction", "segment", "run_time_in_seconds", "length"])

    start_dt = _combine_date_time(df["date"], df["start_time"])
    end_dt = _combine_date_time(df["date"], df["end_time"])
    end_dt = _adjust_overnight(start_dt, end_dt)

    computed_runtime = (end_dt - start_dt).dt.total_seconds()
    df["run_time_in_seconds"] = df["run_time_in_seconds"].where(
        df["run_time_in_seconds"].notna() & (df["run_time_in_seconds"] >= 0),
        computed_runtime,
    )

    df = df.drop_duplicates()
    df, dropped_invalid = _drop_invalid_rows(
        df,
        ["trip_id", "deviceid", "direction", "segment", "date", "start_time", "end_time"],
    )
    df = df[df["run_time_in_seconds"].notna() & (df["run_time_in_seconds"] >= 0)]

    _write_clean_csv(df, "bus_running_times_654_clean.csv")

    return {
        "file": path.name,
        "rows_in": original_rows,
        "rows_out": len(df),
        "rows_dropped_invalid": dropped_invalid,
        "rows_dropped_duplicates": original_rows - len(df) - dropped_invalid,
    }


def clean_bus_stops_and_terminals() -> dict:
    path = DATA_DIR / "bus_stops_and_terminals_654.csv"
    df = pd.read_csv(path)
    original_rows = len(df)

    df["stop_id"] = df["stop_id"].astype(str).str.strip()
    df["route_id"] = df["route_id"].astype(str).str.strip()
    df["direction"] = df["direction"].astype(str).str.strip()
    df["address"] = df["address"].astype(str).str.strip()

    _coerce_numeric(df, ["latitude", "longitude"])

    df = df.drop_duplicates()
    df, dropped_invalid = _drop_invalid_rows(
        df,
        ["stop_id", "route_id", "direction", "latitude", "longitude"],
    )

    _write_clean_csv(df, "bus_stops_and_terminals_654_clean.csv")

    return {
        "file": path.name,
        "rows_in": original_rows,
        "rows_out": len(df),
        "rows_dropped_invalid": dropped_invalid,
        "rows_dropped_duplicates": original_rows - len(df) - dropped_invalid,
    }


def clean_bus_trips() -> dict:
    path = DATA_DIR / "bus_trips_654.csv"
    df = pd.read_csv(path)
    original_rows = len(df)

    df["date"] = _normalize_date(df["date"])
    df["start_time"] = _normalize_time(df["start_time"])
    df["end_time"] = _normalize_time(df["end_time"])

    _coerce_numeric(df, ["trip_id", "deviceid", "direction", "duration_in_mins"])

    start_dt = _combine_date_time(df["date"], df["start_time"])
    end_dt = _combine_date_time(df["date"], df["end_time"])
    end_dt = _adjust_overnight(start_dt, end_dt)

    duration_td = pd.to_timedelta(df.get("duration"), errors="coerce")
    computed_mins = (end_dt - start_dt).dt.total_seconds() / 60.0
    duration_from_td = duration_td.dt.total_seconds() / 60.0

    df["duration_in_mins"] = df["duration_in_mins"].where(
        df["duration_in_mins"].notna() & (df["duration_in_mins"] > 0),
        computed_mins,
    )
    df["duration_in_mins"] = df["duration_in_mins"].where(
        df["duration_in_mins"].notna() & (df["duration_in_mins"] > 0),
        duration_from_td,
    )

    df = df.drop_duplicates()
    df, dropped_invalid = _drop_invalid_rows(
        df,
        ["trip_id", "deviceid", "date", "start_terminal", "end_terminal", "start_time", "end_time"],
    )
    df = df[df["duration_in_mins"].notna() & (df["duration_in_mins"] > 0)]

    _write_clean_csv(df, "bus_trips_654_clean.csv")

    return {
        "file": path.name,
        "rows_in": original_rows,
        "rows_out": len(df),
        "rows_dropped_invalid": dropped_invalid,
        "rows_dropped_duplicates": original_rows - len(df) - dropped_invalid,
    }


def run_preprocessing() -> None:
    reports = [
        clean_bus_dwell_times(),
        clean_bus_running_times(),
        clean_bus_stops_and_terminals(),
        clean_bus_trips(),
    ]

    report_df = pd.DataFrame(reports)
    report_path = OUTPUT_DIR / "preprocessing_report.csv"
    report_df.to_csv(report_path, index=False)

    print("Preprocessing complete. Cleaned files are in:", OUTPUT_DIR)
    print(report_df.to_string(index=False))


if __name__ == "__main__":
    run_preprocessing()
