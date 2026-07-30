#!/bin/bash
# Companion to run_calc.sh, for the convergence scan (QMConvergenceStudy):
# runs TeraChem for every candidate QM region written by
# write_distance_scan_regions() or write_csa_scan_regions().
#
# Assumes each frame directory already has a working tddft.in (the same one
# you use for the single-region calc, e.g. frame0/tddft.in) — for every
# candidate directory under it, this copies that template, points
# "qmindices" at the candidate's own region.qm, and resolves "prmtop"/
# "coordinates" to absolute paths (the candidate directory sits one level
# deeper than the template, so the original relative paths would no longer
# resolve correctly).
#
# Run from the config's dir_root, after write_distance_scan_regions()/
# write_csa_scan_regions() has already created the candidate directories.
#
# Usage:
#   ./run_scan_calc.sh shell    # distance-threshold scan: frame_i/shell_<value>/
#   ./run_scan_calc.sh csa      # CSA-threshold scan:      frame_i/csa_<value>/
#
# If you customized "shell-dir-pattern"/"csa-dir-pattern" in qm_region.json
# away from the "shell_{value}"/"csa_{value}" defaults, update SUBDIR_GLOB
# below to match.

set -uo pipefail

KIND="${1:?Usage: $0 <shell|csa>}"
case "$KIND" in
    shell) SUBDIR_GLOB="shell_*" ;;
    csa)   SUBDIR_GLOB="csa_*" ;;
    *) echo "Usage: $0 <shell|csa>" >&2; exit 1 ;;
esac

TDDFT_OUT="tddft.ref.out"   # must match "tddft_output_name" in qm_region.json

for frame_dir in */; do
    frame_dir="${frame_dir%/}"
    template="$frame_dir/tddft.in"
    [ -f "$template" ] || continue

    prmtop_rel=$(awk '$1=="prmtop"{print $2; exit}' "$template")
    coords_rel=$(awk '$1=="coordinates"{print $2; exit}' "$template")
    prmtop_abs=$(cd "$frame_dir" && realpath "$prmtop_rel")
    coords_abs=$(cd "$frame_dir" && realpath "$coords_rel")

    shopt -s nullglob
    for candidate in "$frame_dir"/$SUBDIR_GLOB/; do
        candidate="${candidate%/}"
        region_file="$candidate/region.qm"
        if [ ! -f "$region_file" ]; then
            echo "[WARN] Skipping $candidate: no region.qm found." >&2
            continue
        fi
        region_abs=$(realpath "$region_file")

        sed \
            -e "s#^prmtop.*#prmtop          $prmtop_abs#" \
            -e "s#^coordinates.*#coordinates     $coords_abs#" \
            -e "s#^qmindices.*#qmindices       $region_abs#" \
            "$template" > "$candidate/tddft.in"

        echo "[RUN] $candidate"
        if ! ( cd "$candidate" && terachem tddft.in 1>"$TDDFT_OUT" 2>tddft.err ); then
            echo "[FAIL] $candidate (see $candidate/tddft.err) — continuing with the rest of the scan" >&2
        fi
    done
    shopt -u nullglob
done
