ml purge
ml Amber/24-CUDA-12.2.1

topfile="3nedFH_sphere_nobox.prmtop"
infile="frame.rst7" #Any geometry of the protein. This is just for extracting atom index
chromophoreIndex=66
residueLastIndex=228 #Chromophore index should be included if the chromophore is the residue with the last index.

# Define your chromophore -> chromophore_list.txt contains atoms index (0-based): these atoms define the minimal QM region
chromList="chromophore_list.txt"

# Generate the reference QM residue list and the reference qm region -> region_ref.qm, residue_list.txt
python ../src/choose_ref_qm.py ${topfile} ${chromophoreIndex} ${residueLastIndex} 4.5 ${chromList} --frames-root ./ 

#Parse partial charges from TeraChem TDDFT output
python ../src/parse_vdd_charges.py region_ref.qm tddft.ref.out --frames-root ./ --scratch-dir scr.tddft.ref --out-ground output_dft_vdd.dat --out-excited output_tddft_vdd.dat

#Run Charge Shift Analysis
residueList="residue_list.txt"
qmref="region_ref.qm"
python ../src/charge_shift_by_residue.py ${topfile} ${residueList} ${chromList} ${qmref} --frame-pattern 'frame*' --ground-file output_dft_vdd.dat --excited-file output_tddft_vdd.dat --score-threshold 0.015
