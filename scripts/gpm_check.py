import argparse
from datetime import UTC, datetime, timedelta
from pathlib import Path


def parse_datetime(dtstr: str) -> datetime:
    dt = datetime.fromisoformat(dtstr)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    else:
        dt = dt.astimezone(UTC)
    return dt


def expected_filename(dt: datetime) -> str:
    # dt: start time of the half-hour window
    dt_end = dt + timedelta(minutes=29, seconds=59)
    s_str = dt.strftime("S%H%M%S")
    e_str = dt_end.strftime("E%H%M%S")
    day_str = dt.strftime("%Y%m%d")
    delta = int(
        (dt - dt.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
        / 60
    )
    # Example: 3B-HHR.MS.MRG.3IMERG.20130501-S000000-E002959.0000.V07B.HDF5
    return f"3B-HHR.MS.MRG.3IMERG.{day_str}-{s_str}-{e_str}.{delta:04d}.V07B.HDF5"


def main():
    parser = argparse.ArgumentParser(
        description="Check existence of GPM HDF5 files for a period."
    )
    parser.add_argument(
        "srcroot", type=Path, help="Root directory containing GPM HDF5 files."
    )
    parser.add_argument(
        "--start",
        type=parse_datetime,
        required=True,
        help="Start datetime (inclusive), e.g., 2013-05-01T00:00:00",
    )
    parser.add_argument(
        "--stop",
        type=parse_datetime,
        required=True,
        help="Stop datetime (exclusive), e.g., 2013-05-02T00:00:00",
    )
    args = parser.parse_args()

    dt = args.start
    missing = []
    while dt < args.stop:
        fname = expected_filename(dt)
        fpath = args.srcroot / fname
        if not fpath.exists():
            print(f"Missing: {fpath} {dt:%j}")
            missing.append(fname)
        dt += timedelta(minutes=30)
    print(f"Checked period: {args.start} to {args.stop}")
    print(f"Total missing files: {len(missing)}")


if __name__ == "__main__":
    main()
