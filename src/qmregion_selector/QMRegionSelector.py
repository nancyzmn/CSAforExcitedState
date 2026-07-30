import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from .qm_utils import get_qm_idx, load_universe
from .io_utils import find_frame_dirs
from .parse_utils import parse_int_list_file
from .schema import ChargeSet, ExcitedState
from .adapters import get_adapter
from .selection import select_bright_state
from .convergence import resolve_scan
from .manifest import RunManifest, hash_atom_indices

try:
    from MDAnalysis.analysis.distances import distance_array
except ImportError as e:
    raise SystemExit(
        "MDAnalysis is required but not found.\n"
        "Install it with: pip install MDAnalysis\n"
        "Then re-run this command."
    ) from e

class QMRegionSelector:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.cfg = self.read_config_json(config_path)
        self.chromophore_atoms = parse_int_list_file(self.cfg["chromophore_atoms_file"])
        self.ground_charges: Dict[Path, ChargeSet] = {}
        self.excited_charges: Dict[Path, ChargeSet] = {}
        self.bright_states: Dict[Path, ExcitedState] = {}
        self.charge_shift: Optional[pd.DataFrame] = None
        self.manifest_results: Dict[str, Any] = {}
        self.validate()
    
    def read_config_json(self, path: str) -> Dict[str, Any]:
        """
        Read the user input config file
        
        Args:
            path (str): path to the user input config file

        Raises:
            If `path` does not exist.

        Returns:
            Dict[str, Any]: parsed JSON contents as a dictionary.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        with p.open("r") as f:
            data = json.load(f)
        return data


    def validate(self) -> None:
        """
        Validate configuration, store all frame directories and frame paths.
        
        Validation rules enforced:
        - `chromophore_resid` must be >= 1.
        - `dist-threshold` must be > 0. May be a single value or a list of
          values to scan (see `QMConvergenceStudy`); every value is checked.
        - `bright-index` must be > 0.
        - `osc-threshold` must be within [0, 1].
        - `root-max` must be >= `bright-index`.
        - `score-threshold` must be within [0, 1]. May be a single value or a
          list of values to scan; every value is checked.
        - `es_code` (default `"terachem"`) must name a registered ES-code adapter.
        - If `resid_last_index` < `chromophore_resid`, a warning is printed (the
        chromophore can still be considered if selected).

        Sets the following instance attributes:
        - `self.frame_dirs`: frame directories in `dir_root`
        - `self.first_frame_path`: path to a geometry that MDAnalysis needs for residue definition
        - `self.adapter`: the `ElectronicStructureAdapter` named by `es_code`
        - `self.charge_scheme`: charge scheme (default `"vdd"`) passed to the adapter

        Raises:
            SystemExit:
                If any configuration value is invalid, or if no matching frame
                directories are found under `dir_root`.
        """
        if self.cfg["chromophore_resid"] < 1:
            raise SystemExit("Chromophore residue must be >= 1.")

        dist_thresholds, _ = resolve_scan(self.cfg["dist-threshold"])
        if any(t < 0 for t in dist_thresholds):
            raise SystemExit("Distance threshold(s) to choose QM region must be > 0.")

        if self.cfg["bright-index"] < 0:
            raise SystemExit("The brightest excited state to choose must be > 0.")

        if self.cfg["osc-threshold"] < 0 or self.cfg["osc-threshold"] > 1:
            raise SystemExit("The oscillator strength threshold must be between 0 and 1")

        if self.cfg["root-max"] < self.cfg["bright-index"]:
            raise SystemExit("The highest excited state considered must be larger than the brightest excited state to choose")

        score_thresholds, _ = resolve_scan(self.cfg["score-threshold"])
        if any(t < 0 or t > 1 for t in score_thresholds):
            raise SystemExit("The CSA score threshold(s) must be between 0 and 1")

        es_code = self.cfg.get("es_code", "terachem")
        try:
            self.adapter = get_adapter(es_code)
        except ValueError as e:
            raise SystemExit(str(e))
        self.charge_scheme = self.cfg.get("charge_scheme", "vdd")

        if self.cfg["resid_last_index"] < self.cfg["chromophore_resid"]:
            print(
                "WARNING: resid_last_index < chromophore_resid; "
                "chromophore still considered if selected."
            )

        frames_root = Path(self.cfg["dir_root"])
        frame_pattern = self.cfg["dir_pattern"]

        frame_dirs = find_frame_dirs(frames_root, frame_pattern)
        if not frame_dirs:
            raise SystemExit(
                f"No '{frame_pattern}' directories found under {frames_root.resolve()}."
            )

        self.frame_dirs = frame_dirs
        print(f"Found {len(frame_dirs)} frame directories.")
        self.first_frame_path = self.frame_dirs[0] / self.cfg["frame_filename"]
    
    def compute_residue_mean_mindist(self) -> Dict[int, float]:
        """
        Compute mean minimum distance (Å) from each residue (1..ResidLastIndex) to chromophore residue across frames.
        
        Returns:
            Dict[int, float]:
                Mapping from residue index `r` to the mean minimum distance (Å) from
                residue `r` to the chromophore residue across frames.
        """
        residues = list(range(1, self.cfg["resid_last_index"] + 1))

        per_res_dists: Dict[int, List[float]] = {r: [] for r in residues}
        for d in self.frame_dirs:
            fpath = d / self.cfg["frame_filename"]
            if not fpath.exists():
                print("WARNING: Missing frame file: %s (skipping)", fpath)
                continue
            u = load_universe(str(self.cfg["topfile"]), str(fpath))
            chromo_positions = u.select_atoms(f"resid {self.cfg['chromophore_resid']}").positions
            # One frame per dir assumed; minimum inter-atomic distance to the chromophore
            for r in residues:
                try:
                    res_positions = u.select_atoms(f"resid {r}").positions
                    val = float(distance_array(chromo_positions, res_positions).min())
                except Exception as e:
                    print(f"ERROR: Failed distance for residue {r} in {fpath}: {e}")
                    val = np.nan
                per_res_dists[r].append(val)

        mean_dists = {}
        for r, arr in per_res_dists.items():
            a = np.array(arr, dtype=float)
            a = a[~np.isnan(a)]
            if a.size == 0:
                print("WARNING: No distances computed for residue %d; setting mean=inf", r)
                mean_dists[r] = float('inf')
            else:
                mean_dists[r] = float(a.mean())
        return mean_dists
    
    def getRefQM(
        self,
        dist_threshold: Optional[float] = None,
        mean_dists: Optional[Dict[int, float]] = None,
    ) -> None:
        """
        Compute the mean minimum distance from each residue to the chromophore across
        frames (via `compute_residue_mean_mindist()`), then selects all residues whose mean distance
        is within the cutoff `dist_threshold` (in Å; defaults to `self.cfg["dist-threshold"]`,
        which must then be a single value — pass an explicit `dist_threshold` to resolve
        one region out of a configured scan list, or use `QMConvergenceStudy` to scan it).
        The chromophore residue `self.cfg["chromophore_resid"]` is always included in the
        selected residue list.

        `mean_dists` can be passed in to reuse an already-computed distance map (e.g. when
        called repeatedly for a threshold scan) instead of recomputing it via MDAnalysis each time.

        Given the selected residues, it calls `get_qm_idx(...)` to obtain the corresponding QM atom
        indices from the provided topology (`self.cfg["topfile"]`) and a reference structure
        (`self.first_frame_path`). The QM region always contains the chromophore atoms set by `chromophore_atoms_file`

        Sets the following instance attributes:
        - `self.qm_ref_residues`: sorted list of selected residue indices (including the chromophore residue index).
        - `self.qm_ref_atoms`: sorted list of QM atom indices for the reference QM region (including the chromophore atoms).
        """
        if dist_threshold is None:
            dist_threshold = self.cfg["dist-threshold"]
            if isinstance(dist_threshold, (list, tuple)):
                raise ValueError(
                    "dist-threshold is a list; pass an explicit dist_threshold to "
                    "resolve a single reference region, or use QMConvergenceStudy to scan it."
                )

        if mean_dists is None:
            mean_dists = self.compute_residue_mean_mindist()
        selected_residues = sorted([r for r, md in mean_dists.items() if md <= dist_threshold])

        if self.cfg["chromophore_resid"] not in selected_residues:
            selected_residues.append(self.cfg["chromophore_resid"])

        qm_from_res = get_qm_idx(
            residues=selected_residues,
            topfile=str(self.cfg["topfile"]),
            infile=str(self.first_frame_path),
            chromophore_resid=self.cfg["chromophore_resid"]
        )

        qm_all_atoms = sorted(set(qm_from_res) | set(self.chromophore_atoms))
        self.qm_ref_atoms = qm_all_atoms
        self.qm_ref_residues = selected_residues

        self.manifest_results["dist_threshold_used"] = dist_threshold
        self.manifest_results["qm_ref_residues"] = self.qm_ref_residues
        self.manifest_results["qm_ref_atoms_count"] = len(self.qm_ref_atoms)
        self.manifest_results["qm_ref_atoms_sha256"] = hash_atom_indices(self.qm_ref_atoms)

    def write_ref_outputs(self) -> None:
        """
        Output: 
            Write the selected QM reference atoms and residues
        """
        out_atoms = Path(self.cfg["out_ref_atoms"])
        out_res = Path(self.cfg["out_ref_residues"])
        np.savetxt(out_atoms, np.array(self.qm_ref_atoms, dtype=int), fmt="%d")
        np.savetxt(out_res, np.array(self.qm_ref_residues, dtype=int), fmt="%d")

        print(f"Selected residues (1-based): {self.qm_ref_residues}")
        print(f"Wrote QM atoms to: {out_atoms}")
        print(f"Wrote residue list to: {out_res}")

    def getGroundCharge(self) -> None:
        """
        Collect and parse ground-state charge files for each frame directory,
        via `self.adapter` (code named by `es_code`, default TeraChem VDD).

        For every directory in `self.frame_dirs`, this method looks for the ground-state charge file at:
            <frame_dir> / self.cfg["scratch-dir"] / self.cfg["ground_charge_file"]

        Output:
            Write per-frame ground-state charge output files at `<frame_dir>/self.cfg["out-ground"]`.

        Sets:
            `self.ground_charges`: Dict[Path, ChargeSet] mapping each frame directory to its parsed ground-state charges.
        """
        for d in self.frame_dirs:
            ground_src = d / self.cfg["scratch-dir"] / self.cfg["ground_charge_file"]
            ground_dst = d / self.cfg["out-ground"]
            if not ground_src.exists():
                print(f"[WARN] Missing ground-state VDD file: {ground_src}")
            else:
                try:
                    charge_set = self.adapter.parse_ground_charges(ground_src, self.charge_scheme)
                    charge_set.to_file(ground_dst)
                    self.ground_charges[d] = charge_set
                    print(f"[OK] Wrote ground VDD → {ground_dst}")
                except Exception as e:
                    print(f"[ERROR] Failed to process ground VDD in {d}: {e}")

    def getExcitedCharge(self) -> None:
        """
        Extract and write excited-state charges for a selected “bright” TDDFT root per frame,
        via `self.adapter` (code named by `es_code`, default TeraChem VDD).

        For each directory in `self.frame_dirs`, this method:
        1) Reads the TDDFT output file `<frame_dir> / self.cfg["tddft_output_name"]`.
        2) Parses the TDDFT “roots table” to obtain oscillator strengths up to `self.cfg["root-max"]`
        3) Filters roots whose oscillator strength is >= `self.cfg["osc-threshold"]` and
        selects the N-th brightest root, where N = `self.cfg["bright-index"]` (1-based).
        If fewer than N roots clear the threshold, that frame is skipped (warning
        printed) and processing continues with the remaining frames.
        4) Parse the charges of this root

        Output:
            Write per-frame excited-state charge output files at `<frame_dir>/self.cfg["out-excited"]`.

        Sets:
            `self.excited_charges`: Dict[Path, ChargeSet] mapping each frame directory to its parsed excited-state charges.
            `self.bright_states`: Dict[Path, ExcitedState] mapping each frame directory to the
                selected bright state (root, excitation energy, oscillator strength) — this is
                the reference-region spectrum a `QMConvergenceStudy` CSA-threshold scan compares
                candidate regions against, with no extra parsing or QM calculation required.
            `self.manifest_results["method"]`/`["basis"]`: level of theory, if the adapter could
                determine it from the ES output (see `write_manifest()`); None otherwise.
        """
        for d in self.frame_dirs:
            tddft_path = d / self.cfg["tddft_output_name"]
            if not tddft_path.exists():
                print(f"[WARN] Missing TDDFT output: {tddft_path}")
                continue

            tddft_text = tddft_path.read_text(errors="ignore")
            states = self.adapter.parse_excited_states(tddft_text, self.cfg["root-max"])
            if not states:
                print("[WARN] Could not find roots table / oscillator strengths.")
                continue
            bright_state = select_bright_state(states, self.cfg["osc-threshold"], self.cfg["bright-index"])
            if bright_state is None:
                print(f"[WARN] No root meets osc-threshold in {d.name}; skipping.")
                continue
            bright_root = bright_state.root
            print(f"Bright state: Root {bright_root}")
            self.bright_states[d] = bright_state

            excited_dst = d / self.cfg['out-excited']
            charge_set = self.adapter.parse_excited_charges(
                tddft_text, bright_root, len(self.qm_ref_atoms), self.charge_scheme
            )
            charge_set.to_file(excited_dst)
            self.excited_charges[d] = charge_set
            print(f"[OK] Wrote excited VDD → {excited_dst}")

        if self.excited_charges:
            first_charge_set = next(iter(self.excited_charges.values()))
            self.manifest_results["method"] = first_charge_set.method
            self.manifest_results["basis"] = first_charge_set.basis
            if first_charge_set.method is None and first_charge_set.basis is None:
                print(
                    "[NOTE] method/basis not recorded for the manifest — auto-extraction "
                    "is currently only implemented for TeraChem output."
                )

    def partition_qm_atoms_by_residues_loo(self) -> Dict[int, List[int]]:
        """
        LOO partition: for each residue r (excluding chromophore), compute atoms that
        vanish from the full QM set when r is removed (chromophore kept).
        
        Returns:
            Dict[int, List[int]]:
                Dictionary mapping each non-chromophore residue index to a sorted list of
                atom indices attributed to that residue under the LOO definition.
        """
        out: Dict[int, List[int]] = {}
        for r in self.qm_ref_residues:
            if r == self.cfg["chromophore_resid"]:
                continue
            # residues-without-r 
            res_wo = self.qm_ref_residues.copy()
            res_wo.remove(r)
            # QM atoms when r is excluded (plus chromophore atoms)
            qm_without_r = set(get_qm_idx(res_wo, 
                                        str(self.cfg["topfile"]), 
                                        str(self.first_frame_path), 
                                        self.cfg["chromophore_resid"])) | set(self.chromophore_atoms)
            # Atoms attributed to r under LOO:
            atoms_for_r = sorted(set(self.qm_ref_atoms) - qm_without_r)
            out[r] = atoms_for_r
        return out

    def getChargeShiftPerResidue(self) -> None:
        """
        Compute per-residue charge shifts (chosen excited state − ground state) over frames and write results to CSV.

        This method quantifies how much atomic charge associated with each (non-chromophore)
        residue changes upon excitation, using precomputed per-atom charge vectors for the QM
        reference region.
        
        Output:
            Write a per-frame, per-residue table to `self.cfg["out-csa-charge-shift"]` with columns:
                `frame`, `resid`, `ground`, `excited`, `delta`
            Write a per-residue summary CSV (mean/std/count of `delta`) named `<out-csa-charge-shift stem>_summary.csv`
        
        Sets the following instance attributes:
            `self.charge_shift`: full per-frame table (pd.DataFrame)
        
        Raises:
            SystemExit:
                If no conformation could be processed
        """
        # Partition QM atoms by residue (excluding chromophore)
        per_res_atoms = self.partition_qm_atoms_by_residues_loo()
        pos_map = {int(atom): pos for pos, atom in enumerate(self.qm_ref_atoms)}
        pre_res_positions: Dict[int, List[int]] = {}
        for r in self.qm_ref_residues:
            if r == self.cfg["chromophore_resid"]:
                continue
            else:
                res_idx = per_res_atoms.get(r, [])
                positions = [pos_map[a] for a in res_idx if a in pos_map]
                pre_res_positions[r] = positions

        records: List[Dict[str, float | int | str]] = []
        for d in self.frame_dirs:
            ground_set = self.ground_charges.get(d)
            excited_set = self.excited_charges.get(d)
            if ground_set is None or excited_set is None:
                g_path = d / self.cfg["out-ground"]
                e_path = d / self.cfg["out-excited"]
                if not g_path.exists() or not e_path.exists():
                    print(f"[WARN] Skipping {d.name}: missing {g_path.name if not g_path.exists() else e_path.name}")
                    continue
                if ground_set is None:
                    ground_set = ChargeSet.from_file(
                        g_path, scheme=self.charge_scheme, state_label="S0", source_code=self.adapter.name
                    )
                if excited_set is None:
                    excited_set = ChargeSet.from_file(
                        e_path, scheme=self.charge_scheme, state_label="excited", source_code=self.adapter.name
                    )

            g = ground_set.charges
            e = excited_set.charges
            if (len(g) < len(self.qm_ref_atoms)) or (len(e) < len(self.qm_ref_atoms)):
                print(f"[WARN] Skipping {d.name}: charge vector shorter than reference QM region "
                    f"(ground={len(g)}, excited={len(e)}, qm={len(self.qm_ref_atoms)})")
                continue
            for r in self.qm_ref_residues:
                if r == self.cfg["chromophore_resid"]:
                    continue
                pos_list = pre_res_positions.get(r, [])
                if not pos_list:
                    print(f"[WARN] Can't find the position in reference qm region for residue {r}")
                    ground = 0.0
                    excited = 0.0
                else:
                    ground = float(g[pos_list].sum())
                    excited = float(e[pos_list].sum())
                delta = excited - ground
                records.append({
                    "frame": d.name,
                    "resid": int(r),
                    "ground": round(ground, 6),
                    "excited": round(excited, 6),
                    "delta": round(delta, 6)
                })
        if not records:
            raise SystemExit("No frames processed; nothing to write.")

        df = pd.DataFrame.from_records(records)
        df["resid"] = pd.Categorical(df["resid"], categories=self.qm_ref_residues, ordered=True)
        df.sort_values(["frame", "resid"], inplace=True)
        df.to_csv(self.cfg["out-csa-charge-shift"], index=False)
        self.charge_shift = df
        print(f"[OK] Wrote {self.cfg['out-csa-charge-shift']} with {len(df)} rows.")

        summary = df.groupby("resid", observed=True)["delta"].agg(["mean", "std", "count"]).reset_index()
        summary_out = Path(self.cfg['out-csa-charge-shift']).stem + "_summary.csv"
        summary.to_csv(summary_out, index=False)
        print(f"[OK] Wrote per-residue summary → {summary_out}")
    
    def compute_csa_selection(self, score_threshold: Optional[float] = None) -> Tuple[pd.DataFrame, List[int]]:
        """
        Score residues by normalized LOO charge shift and select those meeting `score_threshold`
        (defaults to `self.cfg["score-threshold"]`, which must then be a single value — pass an
        explicit `score_threshold` to resolve one selection out of a configured scan list, or use
        `QMConvergenceStudy` to scan it). Uses `self.charge_shift` (computed via
        `getChargeShiftPerResidue()` if not already set), i.e. the charge shifts from the
        reference QM region — scanning `score_threshold` does not require recomputing charges.

        Returns
        -------
        Tuple[pd.DataFrame, List[int]]
            Per-residue score table (mean/std/count of normalized |Δ|, plus a `chosen` column),
            and the selected residue list with the chromophore appended last (matching the
            `residue_list.txt` file convention), not otherwise sorted/deduplicated.
        """
        threshold = score_threshold
        if threshold is None:
            threshold = self.cfg["score-threshold"]
            if isinstance(threshold, (list, tuple)):
                raise ValueError(
                    "score-threshold is a list; pass an explicit score_threshold to "
                    "resolve a single CSA region, or use QMConvergenceStudy to scan it."
                )

        if self.charge_shift is None:
            self.getChargeShiftPerResidue()
        df = self.charge_shift.copy()
        df["abs_delta"] = df["delta"].abs()

        # Per-frame max |Δ| EXCLUDING chromophore
        frame_max = (
            df.groupby("frame", observed=True)["abs_delta"].max().rename("frame_max_nonchrom")
        )

        df = df.merge(frame_max, on="frame", how="left")
        df["denom"] = df["frame_max_nonchrom"].clip(lower=1e-12)
        df["norm_abs_delta"] = df["abs_delta"] / df["denom"]

        # Per-residue score across frames (mean/std of normalized |Δ|)
        score = (df.groupby("resid", observed=True)["norm_abs_delta"]
                .agg(["mean", "std", "count"])
                .reset_index()
                .rename(columns={"mean": "mean_norm_abs_delta",
                                    "std": "std_norm_abs_delta",
                                    "count": "n_frames_used"}))
        # Selection (exclude chrom from thresholding; append it later)
        chosen_mask = (score["mean_norm_abs_delta"] >= threshold)
        score["chosen"] = chosen_mask

        selected = score.loc[score["chosen"], "resid"].astype(int).tolist()
        selected.append(int(self.cfg["chromophore_resid"]))
        return score, selected

    def getCSARegion(self) -> None:
        """
        Score residues and select a "CSA region" using `self.cfg["score-threshold"]` (must be a
        single value; use `QMConvergenceStudy` to scan a list of thresholds instead), via
        `compute_csa_selection()`.

        Output:
            Write three output files: CSA score of all residues, selected residues by CSA, and selected QM atom indices by CSA.
        """
        score, selected = self.compute_csa_selection()

        out_score = self.cfg["out-csa-score"]
        score.to_csv(out_score, index=False)
        print(f"[OK] Wrote score summary → {out_score}")

        np.savetxt(self.cfg["out-selected-residues"], np.array(selected, dtype=int), fmt="%d")
        print(f"[OK] Wrote selected residues → {self.cfg['out-selected-residues']}")

        csa_qmregion = sorted(set(get_qm_idx(selected,
                                    str(self.cfg["topfile"]),
                                    str(self.first_frame_path),
                                    self.cfg["chromophore_resid"])) | set(self.chromophore_atoms))
        np.savetxt(self.cfg["out-selected-qmregion"], np.array(np.unique(csa_qmregion), dtype=int), fmt="%d")
        print(f"[OK] Wrote selected QM region by CSA → {self.cfg['out-selected-qmregion']}")

        self.manifest_results["score_threshold_used"] = self.cfg["score-threshold"]
        self.manifest_results["csa_selected_residues"] = sorted(int(r) for r in selected)
        self.manifest_results["csa_selected_atoms_count"] = len(csa_qmregion)
        self.manifest_results["csa_selected_atoms_sha256"] = hash_atom_indices(csa_qmregion)

    def write_manifest(self) -> None:
        """
        Write a run manifest recording what produced this run's outputs, so
        results stay comparable across systems: code version/git commit, the
        resolved config verbatim, and whatever this run has actually observed
        so far (`self.manifest_results` — the QM region, method/basis, CSA
        selection, ...). Can be called at any point in the pipeline; fields
        for stages that haven't run yet are simply absent.

        Output:
            Write the manifest to `self.cfg.get("out-manifest", "run_manifest.json")`.
        """
        manifest = RunManifest.build(config=self.cfg, results=self.manifest_results)
        out_path = self.cfg.get("out-manifest", "run_manifest.json")
        manifest.to_file(out_path)
        print(f"[OK] Wrote run manifest → {out_path}")
