#! /bin/bash

cp $HOME/experimento.sh $SCRATCH/
cp $HOME/2d.cu $SCRATCH/
cp $HOME/2D.cpp $SCRATCH/
#gpu_control 405 645
[[ $HOSTNAME == "blaise" ]] && gpu_control 715 1075
[[ $HOSTNAME =~ "draco" ]] && gpu_control 324 324


#exit
#nvidia-smi -lgc 810,810 && nvidia-smi -lmc 1911,1911

nvidia-smi --auto-boost-default=DISABLED

 

function experimento(){
	lspci -v > resultados/$4/$HOSTNAME/hardware-lspci
	nvidia-smi > resultados/$4/$HOSTNAME/harwade-nvidia
	lstopo > resultados/$4/$HOSTNAME/harwade-lstopo
	g++ --version > resultados/$4/$HOSTNAME/version-g++
	segundos=$(date +%s)
	
[[ "${1}" == "cuda" ]]  && nvcc 2d.cu -o simulador_2D$HOSTNAME && chmod +x simulador_2D$HOSTNAME && ./simulador_2D$HOSTNAME $3 $4 >> resultados/$4/$HOSTNAME/$1/$3/execucao-$segundos-${HOSTNAME}-exec${2}-threads${3}-cuda.log  &
[[ "${1}" == "opencl" ]] && g++ -o simulador_2d$HOSTNAME 2D.cpp -lOpenCL && chmod +x simulador_2d$HOSTNAME && ./simulador_2d$HOSTNAME $3 $4 >> resultados/$4/$HOSTNAME/$1/$3/execucao-$segundos-${HOSTNAME}-exec${2}-threads${3}-opencl.log &
sleep 1.5
SIMULADOR=$(pgrep simulador_2)

[[ -n $(which nvidia-smi) ]] && placa="nvidia"
[[ -n $(which rocm-smi) ]] && placa="amd"


while [[ -n $SIMULADOR ]]; do
    
    timestamp=$(date -u +"%Y-%m-%d %H:%M:%S.%6N UTC")

    
    [[ "${placa}" == "nvidia" ]] && output=$(nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,clocks.sm,clocks.mem,power.draw --format=csv,noheader,nounits)
    [[ "${placa}" == "amd" ]] && output=$(rocm-smi -a --csv)

    
    echo "${timestamp}, ${output}" >> resultados/$4/$HOSTNAME/$1/$3/registro-$segundos-${HOSTNAME}-${1}-exec${2}-threads${3}.csv
    SIMULADOR=$(pgrep simulador_2)
    
done

}

DIM="1024"
TH="32 64 128 192 256 288 320 384 480 512"

for dim in $DIM
do
for threads in $TH
do
    for execucao in {0..35}
    do
	mkdir -p resultados/$dim/$HOSTNAME/$1/$threads
        experimento $1 $execucao $threads $dim
    done
done
done

