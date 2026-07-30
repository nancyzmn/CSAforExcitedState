from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from .QMRegionSelector import QMRegionSelector
from .qm_utils import get_qm_idx
from .selection import select_bright_state
from .convergence import resolve_scan, find_converged_value
from .spectra import shape_diff


class QMConvergenceStudy:
    """
    Companion to QMRegionSelector/CSA: grows the QM region across a list of
    distance thresholds (before CSA, to pick the reference region) or CSA
    score thresholds (after CSA, to pick the refined region), writing one
    candidate region per value for the user to run externally through their
    ES code, then parses the results and reports where the tracked bright
    state stabilizes.

    Distance-threshold scan: convergence is judged relative to the *next*
    threshold in the scan (there is no reference yet — this step is what
    decides the reference region).

    CSA-threshold scan: convergence is judged relative to the reference
    region's own spectrum, i.e. `selector.bright_states` as populated by
    `QMRegionSelector.getExcitedCharge()` on the (already converged)
    reference region — no extra QM calculation needed for the reference
    side, only for each candidate smaller region.

    Either scan can use a single geometry or an ensemble (typically ~20
    frames), and what "convergence" measures differs between them:
    - Single geometry: the raw excitation energy of that one frame, compared
      against `energy-tol-eV`.
    - Ensemble: each candidate's per-frame (energy, oscillator strength)
      pairs are combined into a Gaussian-broadened spectrum and compared —
      to the next value (distance scan) or the reference (CSA scan) — on its
      peak position (`energy-tol-eV`) and its shape (`shape-tol`); see
      `spectra.shape_diff`. The mean/std of the raw per-frame energies are
      still reported for diagnostics but do not gate convergence here.
    """

    def __init__(self, selector: QMRegionSelector):
        self.selector = selector
        self.cfg = selector.cfg.get("convergence", {})

    @property
    def energy_tol(self) -> float:
        return self.cfg.get("energy-tol-eV", 0.04)

    @property
    def shape_tol(self) -> float:
        return self.cfg.get("shape-tol", 0.02)

    @property
    def sigma(self) -> float:
        return self.cfg.get("sigma-eV", 0.05)

    @property
    def energy_grid(self) -> np.ndarray:
        lo, hi, n = self.cfg.get("energy-grid-eV", [0.5, 4.5, 1000])
        return np.linspace(lo, hi, int(n))

    @property
    def min_stable_points(self) -> int:
        return self.cfg.get("min-stable-points", 2)

    @property
    def is_ensemble(self) -> bool:
        return len(self.selector.frame_dirs) > 1

    # ------------------------------------------------------------------
    # Region generation
    # ------------------------------------------------------------------

    def write_distance_scan_regions(self) -> Dict[float, List[int]]:
        """
        For each value in `selector.cfg["dist-threshold"]` (must be a list),
        write the candidate QM region under `<frame_dir>/<shell-dir-pattern>/`
        for every frame, for the user to run their ES code in.

        Returns
        -------
        Dict[float, List[int]]
            Mapping from distance threshold to its QM atom indices.
        """
        sel = self.selector
        values, is_scan = resolve_scan(sel.cfg["dist-threshold"])
        if not is_scan:
            raise ValueError("dist-threshold is a single value; nothing to scan.")

        mean_dists = sel.compute_residue_mean_mindist()
        pattern = self.cfg.get("shell-dir-pattern", "shell_{value}")

        region_by_value: Dict[float, List[int]] = {}
        for v in values:
            sel.getRefQM(dist_threshold=v, mean_dists=mean_dists)
            region_by_value[v] = list(sel.qm_ref_atoms)
            for d in sel.frame_dirs:
                shell_dir = d / pattern.format(value=v)
                shell_dir.mkdir(parents=True, exist_ok=True)
                np.savetxt(shell_dir / "region.qm", np.array(sel.qm_ref_atoms, dtype=int), fmt="%d")
                np.savetxt(shell_dir / "residues.txt", np.array(sel.qm_ref_residues, dtype=int), fmt="%d")
            print(
                f"[OK] Wrote shell region for dist-threshold={v} "
                f"({len(sel.qm_ref_atoms)} atoms, {len(sel.qm_ref_residues)} residues)"
            )
        return region_by_value

    def write_csa_scan_regions(self) -> Dict[float, List[int]]:
        """
        For each value in `selector.cfg["score-threshold"]` (must be a list),
        write the candidate CSA-refined QM region under
        `<frame_dir>/<csa-dir-pattern>/` for every frame, for the user to run
        their ES code in. Requires `selector.charge_shift` to already be
        computed against the reference region (i.e.
        `getChargeShiftPerResidue()` has run) — scanning score-threshold
        reuses that charge-shift table rather than recomputing charges.

        Returns
        -------
        Dict[float, List[int]]
            Mapping from score threshold to its QM atom indices.
        """
        sel = self.selector
        values, is_scan = resolve_scan(sel.cfg["score-threshold"])
        if not is_scan:
            raise ValueError("score-threshold is a single value; nothing to scan.")
        if sel.charge_shift is None:
            raise RuntimeError(
                "selector.charge_shift is not set; run getChargeShiftPerResidue() "
                "against the reference region before scanning score-threshold."
            )

        pattern = self.cfg.get("csa-dir-pattern", "csa_{value}")
        region_by_value: Dict[float, List[int]] = {}
        for v in values:
            _, selected_residues = sel.compute_csa_selection(score_threshold=v)
            atoms = sorted(
                set(get_qm_idx(selected_residues, str(sel.cfg["topfile"]), str(sel.first_frame_path),
                                sel.cfg["chromophore_resid"]))
                | set(sel.chromophore_atoms)
            )
            region_by_value[v] = atoms
            for d in sel.frame_dirs:
                csa_dir = d / pattern.format(value=v)
                csa_dir.mkdir(parents=True, exist_ok=True)
                np.savetxt(csa_dir / "region.qm", np.array(atoms, dtype=int), fmt="%d")
                np.savetxt(csa_dir / "residues.txt", np.array(selected_residues, dtype=int), fmt="%d")
            print(
                f"[OK] Wrote CSA region for score-threshold={v} "
                f"({len(atoms)} atoms, {len(selected_residues)} residues)"
            )
        return region_by_value

    # ------------------------------------------------------------------
    # Parsing ES output for each scanned value
    # ------------------------------------------------------------------

    def _parse_scan_results(self, values: List[float], dir_pattern: str, scan_label: str) -> pd.DataFrame:
        sel = self.selector
        records = []
        for v in values:
            for d in sel.frame_dirs:
                tddft_path = d / dir_pattern.format(value=v) / sel.cfg["tddft_output_name"]
                if not tddft_path.exists():
                    print(f"[WARN] Missing {scan_label} TDDFT output for value={v}: {tddft_path}")
                    continue

                text = tddft_path.read_text(errors="ignore")
                states = sel.adapter.parse_excited_states(text, sel.cfg["root-max"])
                if not states:
                    print(f"[WARN] Could not find roots table for {scan_label} value={v} in {d.name}.")
                    continue
                bright = select_bright_state(states, sel.cfg["osc-threshold"], sel.cfg["bright-index"])
                if bright is None:
                    print(f"[WARN] No root meets osc-threshold for {scan_label} value={v} in {d.name}; skipping.")
                    continue

                records.append({
                    "value": v,
                    "frame": d.name,
                    "root": bright.root,
                    "excitation_energy_eV": bright.excitation_energy,
                    "oscillator_strength": bright.oscillator_strength,
                })

        if not records:
            raise SystemExit(f"No {scan_label} scan results parsed; nothing to analyze.")
        return pd.DataFrame.from_records(records)

    def parse_distance_scan_results(self) -> pd.DataFrame:
        values, is_scan = resolve_scan(self.selector.cfg["dist-threshold"])
        if not is_scan:
            raise ValueError("dist-threshold is a single value; nothing to scan.")
        pattern = self.cfg.get("shell-dir-pattern", "shell_{value}")
        return self._parse_scan_results(values, pattern, "distance")

    def parse_csa_scan_results(self) -> pd.DataFrame:
        values, is_scan = resolve_scan(self.selector.cfg["score-threshold"])
        if not is_scan:
            raise ValueError("score-threshold is a single value; nothing to scan.")
        pattern = self.cfg.get("csa-dir-pattern", "csa_{value}")
        return self._parse_scan_results(values, pattern, "CSA")

    # ------------------------------------------------------------------
    # Convergence analysis
    # ------------------------------------------------------------------

    @staticmethod
    def _spectrum_inputs(df: pd.DataFrame, value: float) -> Tuple[List[float], List[float]]:
        rows = df[df["value"] == value]
        return rows["excitation_energy_eV"].tolist(), rows["oscillator_strength"].tolist()

    def analyze_distance_scan(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Summarize a distance-threshold scan, judging each value against the
        *next* (larger) threshold — the largest tested value has no next
        value to compare against, so it can never be confirmed converged by
        this method alone.

        What "converged" means depends on how many frames were scanned:
        - Single geometry: the raw excitation energy of that one frame is
          compared directly to the next threshold's (tolerance `energy-tol-eV`).
        - Ensemble (multiple frames, typically ~20): each value's per-frame
          (energy, oscillator strength) pairs are combined into a
          Gaussian-broadened spectrum (`spectra.shape_diff`) and compared to
          the next value's spectrum on two axes — the peak position
          (`delta_spectrum_peak_eV`, tolerance `energy-tol-eV`) and the shape
          (`delta_spectrum_shape`, tolerance `shape-tol`). The mean/std of
          the raw per-frame energies are still reported for reference but do
          not gate convergence in this case.
        """
        values = sorted(df["value"].unique())
        ensemble = self.is_ensemble

        rows = []
        for i, v in enumerate(values):
            rows_v = df[df["value"] == v]
            row = {
                "value": v,
                "mean_excitation_energy_eV": rows_v["excitation_energy_eV"].mean(),
                "std_excitation_energy_eV": rows_v["excitation_energy_eV"].std(),
                "n_frames_used": len(rows_v),
            }

            has_next = i + 1 < len(values)
            if has_next:
                next_mean = df[df["value"] == values[i + 1]]["excitation_energy_eV"].mean()
                row["delta_energy_eV"] = next_mean - row["mean_excitation_energy_eV"]
            else:
                row["delta_energy_eV"] = np.nan

            if not ensemble:
                # Single geometry: convergence is judged directly on that one
                # frame's excitation energy relative to the next threshold.
                row["within_tol"] = has_next and abs(row["delta_energy_eV"]) <= self.energy_tol
            elif has_next:
                # Ensemble: convergence is judged on the *spectrum* built from
                # every frame's (energy, oscillator strength) pair — its peak
                # position (vs. energy-tol-eV) and its shape (vs. shape-tol) —
                # not on the mean of the raw per-frame energies above.
                e_i, o_i = self._spectrum_inputs(df, v)
                e_j, o_j = self._spectrum_inputs(df, values[i + 1])
                shape, peak_shift = shape_diff(e_i, o_i, e_j, o_j, self.energy_grid, self.sigma)
                row["delta_spectrum_peak_eV"] = peak_shift
                row["delta_spectrum_shape"] = shape
                peak_ok = abs(peak_shift) <= self.energy_tol
                shape_ok = abs(shape) <= self.shape_tol
                row["within_tol"] = peak_ok and shape_ok
            else:
                row["delta_spectrum_peak_eV"] = np.nan
                row["delta_spectrum_shape"] = np.nan
                row["within_tol"] = False  # cannot confirm convergence without a next value to compare to

            rows.append(row)

        summary = pd.DataFrame(rows)
        ok_series = pd.Series(summary["within_tol"].to_numpy(), index=summary["value"].to_numpy())
        converged = find_converged_value(values, ok_series, self.min_stable_points)
        summary["converged_candidate"] = summary["value"] == converged

        if converged is not None:
            print(f"[OK] Distance-threshold scan converged at dist-threshold={converged}")
        else:
            print(
                "[WARN] Distance-threshold scan did not stabilize within the tested range "
                "(or the largest value tested cannot yet be confirmed — try a larger threshold)."
            )
        return summary

    def finalize_reference_region(self, summary: Optional[pd.DataFrame] = None) -> float:
        """
        Resolve the converged dist-threshold from a distance-threshold scan and
        generate + write the final reference QM region for it, via
        `selector.getRefQM()`/`selector.write_ref_outputs()` — the step that
        turns a converged scan into the reference region the rest of the CSA
        pipeline (and, later, the CSA-threshold scan) builds on.

        Parameters
        ----------
        summary
            Result of `analyze_distance_scan()`. If not given, the scan is
            parsed and analyzed here (`parse_distance_scan_results()` +
            `analyze_distance_scan()`).

        Returns
        -------
        float
            The converged dist-threshold value used to generate the region.

        Raises
        ------
        RuntimeError
            If the scan did not converge within the tested range.
        """
        if summary is None:
            summary = self.analyze_distance_scan(self.parse_distance_scan_results())

        converged_values = summary.loc[summary["converged_candidate"], "value"]
        if converged_values.empty:
            raise RuntimeError(
                "Distance-threshold scan did not converge; cannot finalize a reference "
                "region. Try scanning larger threshold values."
            )
        converged_value = float(converged_values.iloc[0])

        self.selector.getRefQM(dist_threshold=converged_value)
        self.selector.write_ref_outputs()
        print(f"[OK] Finalized reference QM region at dist-threshold={converged_value}")
        return converged_value

    def analyze_csa_scan(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Summarize a CSA-threshold scan, judging each value against the
        reference region's own spectrum (`selector.bright_states`, from the
        reference-region QM calculation already required by the main CSA
        pipeline — no extra calculation needed for the reference side).
        Values are ordered from most stringent (fewest residues) to least, so
        the value returned as converged is the smallest region that already
        matches the reference and stays matching as the region is relaxed
        further.

        What "converged" means depends on how many frames were scanned:
        - Single geometry: the raw excitation energy of that one frame is
          compared directly to the reference's (tolerance `energy-tol-eV`).
        - Ensemble (multiple frames, typically ~20): each value's per-frame
          (energy, oscillator strength) pairs are combined into a
          Gaussian-broadened spectrum (`spectra.shape_diff`) and compared to
          the reference spectrum on two axes — the peak position
          (`delta_spectrum_peak_eV`, tolerance `energy-tol-eV`) and the shape
          (`delta_spectrum_shape`, tolerance `shape-tol`). The mean/std of
          the raw per-frame energies are still reported for reference but do
          not gate convergence in this case.
        """
        sel = self.selector
        if not sel.bright_states:
            raise RuntimeError(
                "selector.bright_states is empty; run getExcitedCharge() on the "
                "reference region before scanning score-threshold."
            )
        ref_energies = [s.excitation_energy for s in sel.bright_states.values()]
        ref_osc = [s.oscillator_strength for s in sel.bright_states.values()]
        ref_mean_energy = float(np.mean(ref_energies))

        # Most-stringent (highest score-threshold / smallest region) first.
        values = sorted(df["value"].unique(), reverse=True)
        ensemble = self.is_ensemble

        rows = []
        for v in values:
            rows_v = df[df["value"] == v]
            mean_e = rows_v["excitation_energy_eV"].mean()
            row = {
                "value": v,
                "mean_excitation_energy_eV": mean_e,
                "std_excitation_energy_eV": rows_v["excitation_energy_eV"].std(),
                "n_frames_used": len(rows_v),
                "delta_energy_eV": mean_e - ref_mean_energy,
            }

            if not ensemble:
                # Single geometry: convergence is judged directly on that one
                # frame's excitation energy relative to the reference.
                row["within_tol"] = abs(row["delta_energy_eV"]) <= self.energy_tol
            else:
                # Ensemble: convergence is judged on the *spectrum* built from
                # every frame's (energy, oscillator strength) pair — its peak
                # position (vs. energy-tol-eV) and its shape (vs. shape-tol) —
                # not on the mean of the raw per-frame energies above.
                e_v, o_v = rows_v["excitation_energy_eV"].tolist(), rows_v["oscillator_strength"].tolist()
                shape, peak_shift = shape_diff(ref_energies, ref_osc, e_v, o_v, self.energy_grid, self.sigma)
                row["delta_spectrum_peak_eV"] = peak_shift
                row["delta_spectrum_shape"] = shape
                peak_ok = abs(peak_shift) <= self.energy_tol
                shape_ok = abs(shape) <= self.shape_tol
                row["within_tol"] = peak_ok and shape_ok

            rows.append(row)

        summary = pd.DataFrame(rows)
        ok_series = pd.Series(summary["within_tol"].to_numpy(), index=summary["value"].to_numpy())
        converged = find_converged_value(values, ok_series, self.min_stable_points)
        summary["converged_candidate"] = summary["value"] == converged

        if converged is not None:
            print(f"[OK] CSA-threshold scan converged (vs. reference spectrum) at score-threshold={converged}")
        else:
            print(
                "[WARN] CSA-threshold scan did not match the reference spectrum within "
                "tolerance across the tested range."
            )
        return summary
