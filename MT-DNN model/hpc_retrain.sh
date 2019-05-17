#!/bin/sh
#
#SBATCH --verbose
#SBATCH --job-name=nlu
#SBATCH --output=./out/job_retrain_%j.out
#SBATCH --error=./out/job_retrain_%j.err
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --mem=6GB
#SBATCH --gres=gpu:1

/bin/hostname
/bin/pwd


module purge
module load python3/intel/3.6.3
source ~/pyenv/py3.6.3/bin/activate

cd scripts
./snli_domain_adaptation_bash.sh


