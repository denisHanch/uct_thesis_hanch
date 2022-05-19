#!/bin/bash

#PBS -q cheminf
#PBS -l select=1:ncpus=6:mem=12gb
#PBS -l walltime=72:00:00
#PBS -M hanchard@vscht.cz
#PBS -m ae

set -x

module load lich/cuda-11.1

MODEL=$MODEL
JOBS_INFO="jobs_info_bm_full_$MODEL.txt"

DATADIR=/home/$USER/benchmarking
SCRATCHDIR=/scratch/$USER/$PBS_JOBID
SCRATCH_OUT="output"
OUT_DIR="jobs_output"
OUT_JOB="bench_batch_FULL_${MODEL}_${PREFIX}_$PBS_JOBID"

#ENV_PATH=env.yml
ENV_NAME=aizynth-env

mkdir $DATADIR/$OUT_DIR/$OUT_JOB
mkdir $SCRATCHDIR

cp -r $DATADIR/* $SCRATCHDIR
cd $SCRATCHDIR

mkdir $SCRATCH_OUT

echo "Start at $(date +"%T %d.%m.%Y")" > $PBS_O_WORKDIR/$JOBS_INFO
echo "$PBS_JOBID is running on node `hostname -f` in directory $SCRATCHDIR" >> $PBS_O_WORKDIR/$JOBS_INFO

eval "$(conda shell.bash hook)"

conda activate $PBS_O_WORKDIR/.conda/envs/$ENV_NAME

#python python_scripts/create_fps_reaxys.py
#python python_scripts/pca_reaxys.py
#python python_scripts/balancing.py
#python python_scripts/$MODEL.py
python python_scripts/sep_test_batch/one_for_all_$SCRIPT.py $PREFIX $MODEL

cp -r *.log $SCRATCH_OUT $DATADIR/$OUT_DIR/$OUT_JOB

rm -r $SCRATCHDIR