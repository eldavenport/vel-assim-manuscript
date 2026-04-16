"""
Download HYCOM GLBv0.08/expt_53.X reanalysis (SSH, U, V) for the equatorial Pacific.
Called by hycom_offline_download.sh — do not run directly.

Note: the HYCOM dataset uses -180 to 180 longitude, not 0-360 as documented.
Notebook lon 180-260°E maps to -180 to -100° in the dataset's convention.
"""
import sys
import calendar
from datetime import date
from pathlib import Path

import xarray as xr
from tqdm import tqdm

# ── Parameters (from hycom_reanalysis_clean.ipynb) ──────────────────────────
OUTPUT_DIR    = Path(sys.argv[1])
# Notebook: 180–260°E (0–360 convention) → -180 to -100° (HYCOM -180 to 180)
LON_MIN       = -180.0
LON_MAX       = -100.0
LAT_MIN       = -10.0
LAT_MAX       = 10.0
DEPTH_MIN_POS = 0.0
DEPTH_MAX_POS = 500.0
VARIABLES     = ["surf_el", "water_u", "water_v"]
DEPTH_VARS    = {"water_u", "water_v"}
TIME_START    = date(2013, 5, 1)
TIME_END      = date(2013, 12, 31)
THREDDS_BASE  = "https://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X/data/{year}"

# ── Build monthly date ranges ────────────────────────────────────────────────
def monthly_ranges(start, end):
    ranges = []
    current = start.replace(day=1)
    while current <= end:
        days = calendar.monthrange(current.year, current.month)[1]
        month_end = current.replace(day=days)
        ranges.append((max(current, start), min(month_end, end)))
        current = date(current.year + 1, 1, 1) if current.month == 12 \
                  else date(current.year, current.month + 1, 1)
    return ranges

# ── Download ─────────────────────────────────────────────────────────────────
depth_vars   = [v for v in VARIABLES if v in DEPTH_VARS]
nodepth_vars = [v for v in VARIABLES if v not in DEPTH_VARS]
months       = monthly_ranges(TIME_START, TIME_END)

open_datasets = {}
try:
    with tqdm(months, unit="month", ncols=72) as pbar:
        for month_start, month_end in pbar:
            fname = f"hycom_{month_start.year}_{month_start.month:02d}.nc"
            fpath = OUTPUT_DIR / fname
            pbar.set_description(fname)

            if fpath.exists():
                tqdm.write(f"  Skipping {fname} (already exists)")
                continue

            year = month_start.year
            if year not in open_datasets:
                url = THREDDS_BASE.format(year=year)
                tqdm.write(f"  Opening OPeNDAP dataset for {year} …")
                open_datasets[year] = xr.open_dataset(url, engine="netcdf4",
                                                       drop_variables=["tau"])

            ds = open_datasets[year]
            tslice = slice(month_start.strftime("%Y-%m-%d"),
                           month_end.strftime("%Y-%m-%d"))
            spatial = dict(time=tslice, lat=slice(LAT_MIN, LAT_MAX),
                           lon=slice(LON_MIN, LON_MAX))

            parts = []
            if depth_vars:
                parts.append(ds[depth_vars].sel(
                    **spatial, depth=slice(DEPTH_MIN_POS, DEPTH_MAX_POS)))
            if nodepth_vars:
                parts.append(ds[nodepth_vars].sel(**spatial))

            subset = xr.merge(parts) if len(parts) > 1 else parts[0]
            subset.load()
            subset.to_netcdf(str(fpath))
            tqdm.write(f"  Saved {fname}  ({fpath.stat().st_size / 1e6:.1f} MB)")

finally:
    for ds in open_datasets.values():
        ds.close()
