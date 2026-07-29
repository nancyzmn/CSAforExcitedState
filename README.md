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

Two more keys are accepted but optional, both defaulting to today's only
supported values so existing configs (like the one above) don't need any
changes:
- `"es_code"` (default `"terachem"`): which `ElectronicStructureAdapter` to
  parse ES output with.
- `"charge_scheme"` (default `"vdd"`): which charge partitioning scheme to
  request from that adapter.

## Installation

This repo uses [mamba](https://mamba.readthedocs.io/) to manage its
environment, since `pytraj` (needed for the Amber/MM side) comes from
AMBERTools rather than PyPI:

```bash
mamba env create -f environment.yml
mamba activate qmregion-selector
```

This installs AMBERTools (providing `pytraj`), `numpy`, `pandas`, `pytest`,
and the package itself (editable).

If you're on an HPC system where you can't create your own conda/mamba
environments, an already-installed Amber module may work instead, e.g.:

```bash
module load Amber/24-CUDA-12.2.1
```

AMBER prmtop + per-frame coordinates readable by pytraj (e.g., ```rst7```)

ES code outputs: excited state calculation output, e.g. TeraChem TDDFT. The
ground state atomic charges are usually in the scratch folder.

## File conventions

Residue indices (e.g., residue_list.txt): 1-based, chromophore resid last.

Atom indices (chromophore_atoms.txt, region_ref.qm, etc.): 0-based.

Frame directories typically look like frame1/, frame2/, … each containing a single coordinate file (default frame.rst7) and TeraChem outputs.

## Examples
In the example folder, we use charge shift analysis to find the QM region for mRouge (a red fluorescent protein) based on five geometries. 
