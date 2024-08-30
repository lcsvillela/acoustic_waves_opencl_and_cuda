
[[ $1 == "cuda" ]] && MACHINES="blaise tupi draco draco beagle marcs poti"
[[ $1 == "opencl" ]] && MACHINES="blaise tupi draco draco beagle marcs poti sirius lunaris"


for i in $MACHINES
do

sbatch <<< "#!/bin/bash
#SBATCH --job-name=experimento-$i-$1
#SBATCH --partition=$i
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=23:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

./experimento.sh $1"
done

