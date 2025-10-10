for i in {1..4}
do 
cd frame$i
terachem tddft.in 1>tddft.ref.out 2>tddft.err 
cd ../
done
