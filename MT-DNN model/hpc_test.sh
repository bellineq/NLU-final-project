#!/bin/sh
#
#SBATCH --verbose
#SBATCH --job-name=nlu
#SBATCH --output=./out/job_testMNLI_%j.out
#SBATCH --error=./out/job_testMNLI_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --mem=6GB
#SBATCH --gres=gpu:1
 
/bin/hostname
/bin/pwd
 

module purge
module load python3/intel/3.6.3
source ~/pyenv/py3.6.3/bin/activate

# python test.py --batch_size 2 --batch_size_eval 2 --train_datasets 'snli' --test_datasets 'snli' --output_dir 'testout/baseline' --init_checkpoint 'testmodel/baseline/model.pt'
python testMNLI.py --batch_size 2 --batch_size_eval 2 --output_dir 'testout/MNLI' 

