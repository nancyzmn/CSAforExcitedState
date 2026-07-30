# QMRegion Selector

This repo is designed for workflows like:
1) build a reference QM region based on proximity to the chromophore, then  
2) identify residues with the largest charge shift upon excitation (Δcharge per residue) and select them for an updated QM region.

---

## What it does

### 1) Reference QM region by distance
- Reads an Amber topology (`.prmtop`) and per-frame coordinates (e.g., `frame.rst7`)
- Uses a **chromophore atom list** to define the chromophore (your minimal QM region)
- Selects residues whose atoms fall within a distance threshold of the chromophore (e.g., 4.5 Å)
- Writes:
  - a QM region atom list (e.g., `.qm`)
  - a residue list text file

### 2) Charge Shift Analysis
If you provide electronic structure (ES) code output and charge files, QMRegion Selector can:
- Parse ground-state charges and excited-state charges (for selected bright root)
- Compute per-residue charge shifts (Δq = q_excited − q_ground, aggregated per residue)
- Score residues and pick those exceeding a score threshold
- Write CSA summary tables and a refined QM region definition

Parsing is done through a small adapter interface
(`qmregion_selector.adapters.ElectronicStructureAdapter`), so CSA isn't tied to
one ES code. TeraChem/VDD is the only adapter shipped today
(`qmregion_selector.adapters.TeraChemAdapter`); adding support for another
code (Q-Chem, ORCA, Psi4, ...) means writing a class that implements
`parse_ground_charges`, `parse_excited_states`, and `parse_excited_charges`,
returning the code-agnostic `ChargeSet`/`ExcitedState` types from
`qmregion_selector.schema`, and registering it with `@register_adapter`.
Select which adapter to use via the `es_code` config key (see below).

### 3) QM-region convergence study
`dist-threshold` (step 1) and `score-threshold` (step 2) each normally take a
single number. Give either one a **list** instead, and
`qmregion_selector.QMConvergenceStudy` will scan it: it writes one candidate
QM region per value (for you to run through your ES code, same as any other
region), then parses the results and reports where the tracked bright state
stabilizes as the region grows.

The two scans answer different questions, so their convergence is judged
against different comparison targets:
- **Distance-threshold scan** (run before CSA, to pick the reference
  region): each value is compared to the *next* threshold in the list —
  "does growing the region further still change anything?" Candidate
  regions are written to `<frame_dir>/shell_<value>/`.
- **CSA-threshold scan** (run after CSA, to pick the refined region): each
  value is compared to the *reference region's own spectrum* (already
  available for free from the reference-region calculation your CSA run
  already required — no extra QM calculation needed for the reference side)
  — "does this smaller region still reproduce the reference?" Candidate
  regions are written to `<frame_dir>/csa_<value>/`.

What "converged" means also depends on whether you scanned one geometry or
an ensemble:
- **Single geometry**: compare that one frame's raw excitation energy
  directly to the comparison target (tolerance `energy-tol-eV`).
- **Ensemble** (multiple frames, typically ~20): every frame's (excitation
  energy, oscillator strength) pair for a given value is combined into a
  Gaussian-broadened absorption spectrum, which is then compared to the
  comparison target's spectrum on two axes — the shift in the spectrum's
  peak position (tolerance `energy-tol-eV`) and the difference in the
  spectrum's shape (tolerance `shape-tol`), both from `spectra.shape_diff`.
  The mean/std of the raw per-frame energies are still reported per value
  for reference, but the ensemble case is gated on the spectrum, not on
  that mean.

```python
from qmregion_selector import QMRegionSelector, QMConvergenceStudy

selector = QMRegionSelector("qm_region.json")   # "dist-threshold": [3.0, 4.5, 6.0, 8.0, 10.0]
study = QMConvergenceStudy(selector)
study.write_distance_scan_regions()
# ... run your ES code in frame_i/shell_<value>/ for every value ...
study.finalize_reference_region()    # resolves the converged threshold and calls
                                      # selector.getRefQM()/write_ref_outputs() for it

selector.getGroundCharge()
selector.getExcitedCharge()          # also becomes the CSA scan's reference spectrum
selector.getChargeShiftPerResidue()

# "score-threshold": [0.005, 0.010, 0.015, 0.020, 0.030]
study.write_csa_scan_regions()
# ... run your ES code in frame_i/csa_<value>/ for every value ...
summary = study.analyze_csa_scan(study.parse_csa_scan_results())
```

Since this repo never launches your ES code for you (same as the single-region
workflow), `example/run_scan_calc.sh` is a batch-runner helper for the "run
your ES code in every candidate directory" steps above — the scan
counterpart to `example/run_calc.sh`. It reuses each frame's existing
`tddft.in`, pointed at that candidate's own `region.qm`:

```bash
./run_scan_calc.sh shell   # after write_distance_scan_regions()
./run_scan_calc.sh csa     # after write_csa_scan_regions()
```

Convergence tolerances (and the Gaussian broadening width used for spectral
shape) live under an optional `"convergence"` config block; all keys default
if omitted:

```json
"convergence": {
  "energy-tol-eV": 0.04,
  "shape-tol": 0.02,
  "sigma-eV": 0.05,
  "energy-grid-eV": [0.5, 4.5, 1000],
  "min-stable-points": 2,
  "shell-dir-pattern": "shell_{value}",
  "csa-dir-pattern": "csa_{value}"
}
```

A single geometry (one frame directory) is a valid input to either scan —
convergence is then judged on excitation energy alone, since spectral shape
needs an ensemble of frames to be meaningful.

### 4) Run manifest / provenance
`selector.write_manifest()` writes a small JSON record of what produced a
run's outputs, so results stay comparable across systems and reproducible
later. It has two parts:
- `config`: the resolved config dict, verbatim — everything that was *asked
  for* (topology, thresholds, chromophore residue, ...), unmodified.
- `results`: what was *actually* computed/observed — the resolved QM region
  (residues, atom count, and a hash of the atom list, so two manifests can be
  compared for an exact-region match without embedding the full list), the
  method/basis TeraChem reported (see below), and the CSA selection if
  `getCSARegion()` ran.

Also records `qmregion_selector_version` (the installed package version) and,
best-effort, `git_commit`/`git_dirty` if the code is running from a git
checkout — `None` for both if not (e.g. installed from a tarball on an HPC
system).

`write_manifest()` can be called at any point in the pipeline — call it early
for partial provenance, or after `getCSARegion()` for the full picture:

```python
selector.getRefQM()
selector.write_ref_outputs()
selector.getGroundCharge()
selector.getExcitedCharge()
selector.getChargeShiftPerResidue()
selector.getCSARegion()
selector.write_manifest()   # -> "run_manifest.json" (or "out-manifest" in config)
```

Method/basis are extracted automatically from TeraChem's output (it prints
`Method: wPBE` / `Using basis set: 6-31gss` once per job) rather than typed
into the config, so they can't drift out of sync with what was actually run.
This is currently TeraChem-specific — `getExcitedCharge()` prints a `[NOTE]`
if it can't determine them, which is expected for other ES-code adapters
until they add the same extraction.

---

## Input configuration (JSON)

The main entrypoint expects a JSON config like:

`example/qm_region.json`
```json
{
  "topfile": "3nedFH_sphere_nobox.prmtop",
  "chromophore_resid": 66,
  "resid_last_index": 228,
  "dist-threshold": 4.5,
  "chromophore_atoms_file": "chromophore_list.txt",

  "dir_pattern": "frame*",
  "frame_filename": "frame.rst7",
  "dir_root": ".",

  "out_ref_atoms": "region_ref_test.qm",
  "out_ref_residues": "residue_list_test.txt",

  "_comment": "The following part is only needed if you want to run CSA",
  "tddft_output_name": "tddft.ref.out",
  "bright-index": 1,
  "scratch-dir": "scr.tddft.ref",
  "ground_charge_file": "charge_vdd.xls",
  "osc-threshold": 0.80,
  "root-max": 6,
  "out-ground": "output_dft_vdd_test.dat",
  "out-excited": "output_tddft_vdd_test.dat",

  "score-threshold": 0.015,
  "out-csa-charge-shift": "charge_shift_by_residue_test.csv",
  "out-csa-score": "csa_score_summary_test.csv",
  "out-selected-residues": "residue_list_csa_test.txt",
  "out-selected-qmregion": "region_CSA_test.qm",

  "out-manifest": "run_manifest_test.json"
}
```

Two more keys are accepted but optional, both defaulting to today's only
supported values so existing configs (like the one above) don't need any
changes:
- `"es_code"` (default `"terachem"`): which `ElectronicStructureAdapter` to
  parse ES output with.
- `"charge_scheme"` (default `"vdd"`): which charge partitioning scheme to
  request from that adapter.

`"dist-threshold"` and `"score-threshold"` also each accept a list instead of
a single number, to run a convergence scan instead of resolving one region —
see "3) QM-region convergence study" above.

`"out-manifest"` (default `"run_manifest.json"`): where `write_manifest()`
writes its provenance record — see "4) Run manifest / provenance" above.

## Installation

Requires Python >= 3.10. The MM/topology side is handled by
[MDAnalysis](https://www.mdanalysis.org/) (not tied to Amber specifically —
any topology/trajectory format MDAnalysis supports should work, though only
Amber prmtop + `rst7` is tested here), which is a plain PyPI package, so a
standard pip install is enough:

```bash
pip install -e .
```

If you'd rather use [mamba](https://mamba.readthedocs.io/)/conda (e.g. to
also get `pytest` and pin the Python version in one step):

```bash
mamba env create -f environment.yml
mamba activate qmregion-selector
```

AMBER prmtop + per-frame coordinates (e.g., ```rst7```)

ES code outputs: excited state calculation output, e.g. TeraChem TDDFT. The
ground state atomic charges are usually in the scratch folder.

## File conventions

Residue indices (e.g., residue_list.txt): 1-based, chromophore resid last.

Atom indices (chromophore_atoms.txt, region_ref.qm, etc.): 0-based.

Frame directories typically look like frame1/, frame2/, … each containing a single coordinate file (default frame.rst7) and TeraChem outputs.

## Examples
In the example folder, we use charge shift analysis to find the QM region for mRouge (a red fluorescent protein) based on five geometries. 
