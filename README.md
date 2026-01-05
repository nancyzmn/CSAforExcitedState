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
If you provide TeraChem outputs and charge files, QMRegion Selector can:
- Parse ground-state VDD charges and excited-state VDD charges (for selected bright root)
- Compute per-residue charge shifts (Δq = q_excited − q_ground, aggregated per residue)
- Score residues and pick those exceeding a score threshold
- Write CSA summary tables and a refined QM region definition

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
  "out-selected-qmregion": "region_CSA_test.qm"
}
```

## Requirements
Python 3.8+: ```pytraj```, ```numpy```, ```pandas```

AMBERTools / `pytraj` (required at runtime)
This package uses `pytraj`, which is typically provided by **AMBERTools** on HPC systems.  
`pytraj` is **not installed via pip** by default here—this project assumes you have AMBERTools available and loaded.

On a cluster/module system, load AMBER before running:

```bash
module load AMBER
```

AMBER prmtop + per-frame coordinates readable by pytraj (e.g., ```rst7```)

TeraChem outputs: excited state calculations output. The ground state atomic charges are usually in the scratch folder.

## File conventions

Residue indices (e.g., residue_list.txt): 1-based, chromophore resid last.

Atom indices (chromophore_atoms.txt, region_ref.qm, etc.): 0-based.

Frame directories typically look like frame1/, frame2/, … each containing a single coordinate file (default frame.rst7) and TeraChem outputs.

## Examples
In the example folder, we use charge shift analysis to find the QM region for mRouge (a red fluorescent protein) based on five geometries. 