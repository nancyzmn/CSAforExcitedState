# CSAforExcitedState
Determine QM residues for QM/MM absorption spectra calculations

**What’s inside**

```qm_utils.py```

```get_qm_idx(residues, topfile, infile, chromophore_resid)```

Build QM atom indices (0-based) from a residue list (1-based). Make sure all QM/MM cuts are at C-C or C-H bonds

```partition_qm_atoms_by_residues_loo(...)```

Leave-one-out (LOO) partition of the current QM region into per-residue atom sets: atoms “belonging” to residue r are those that disappear from the QM set when r is removed (chromophore atoms kept).

```choose_qm_region.py```

Picks residues for a reference QM region using mean min-distance (Å) from each residue to the chromophore across frames. Writes region_ref.qm and residue_list.txt.

```parse_vdd_charges.py```

From each frame directory, extracts:
ground-state VDD charges → output_dft_vdd.dat
excited-state VDD charges for the N-th bright root (by oscillator strength ≥ threshold) → output_tddft_vdd.dat
**Note that this is used to extract information from TeraChem output. Substitute the functions if you're using other electronic structure codes.**

```charge_shift_by_residue.py```

Sums per-residue charges (ground/excited), computes Δ = exc − grd per frame, and optionally scores residues via normalized |Δ| (normalization per frame uses the max over non-chromophore residues). Can write a selected residue list and a CSA-refined QM region.
