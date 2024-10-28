#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=10:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=generate_audios

#SBATCH --output=logs/generate_audios_%j.out
#SBATCH --requeue
#SBATCH --signal=SIGUSR1@90
#SBATCH --mail-user=tk3309@nyu.edu
#SBATCH --mail-type=ALL
#SBATCH --error=logs/generate_audios_error_%j.err
singularity exec --nv \
	    --overlay /scratch/tk3309/LLDM/lldm.ext3:ro \
	    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
	    /bin/bash -c "source /ext3/env.sh; source /scratch/tk3309/stable/bin/activate; python /scratch/tk3309/stable-audio-tools/generated_audio/ddp.py"