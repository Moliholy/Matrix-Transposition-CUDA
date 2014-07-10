#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH -A kurs2014-1-124

increment=1000
n=10000
maximum=100000

while [[ $n -le $maximum ]]
do

./a.out $n $n >>transpose.csv

n=$(( $n+$increment ))

done
