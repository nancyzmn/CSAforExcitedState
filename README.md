# CSAforExcitedState
Determine QM residues for QM/MM absorption spectra calculations

## What’s inside

Need to fill

## Requirements
Python 3.8+: ```pytraj```, ```numpy```, ```pandas```

AMBER prmtop + per-frame coordinates readable by pytraj (e.g., ```rst7```)

TeraChem outputs: excited state calculations output. The ground state atomic charges are usually in the scratch folder.

## File conventions

Residue indices (e.g., residue_list.txt): 1-based, chromophore resid last.

Atom indices (chromophore_atoms.txt, region_ref.qm, etc.): 0-based.

Frame directories typically look like frame1/, frame2/, … each containing a single coordinate file (default frame.rst7) and TeraChem outputs.

## Examples
In the example folder, we use charge shift analysis to find the QM region for mRouge (a red fluorescent protein) based on five geometries. 
