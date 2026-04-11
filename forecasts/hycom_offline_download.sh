#!/usr/bin/env bash
# Download HYCOM GLBv0.08/expt_53.X reanalysis (SSH, U, V) for the equatorial Pacific.
#
# Parameters match hycom_reanalysis_clean.ipynb exactly:
#   Longitude  : 180–260 °E
#   Latitude   : 10 °S – 10 °N
#   Depth      : 0–500 m
#   Variables  : SSH (surf_el), U (water_u), V (water_v)
#   Period     : 01 Sep 2012 – 30 Jun 2013
#   Output     : /data/SO3/edavenport/hycom_data/
#
# Uses the tpose conda environment (netCDF4 with OPeNDAP support).
# Existing monthly files are skipped so interrupted runs can be resumed.

set -euo pipefail

OUTPUT_DIR="/data/SO3/edavenport/tpose6/hycom_data"
mkdir -p "$OUTPUT_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
conda run -n tpose python "$SCRIPT_DIR/hycom_offline_download.py" "$OUTPUT_DIR"

echo "Done."