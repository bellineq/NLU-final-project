#!/bin/sh
#
#SBATCH --verbose
#SBATCH --job-name=nlu
#SBATCH --output=./out/job_train_%j.out
#SBATCH --error=./out/job_train_%j.err
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1
 
/bin/hostname
/bin/pwd
 

module purge
module load python3/intel/3.6.3
source ~/pyenv/py3.6.3/bin/activate

# python train.py --batch_size 2 --batch_size_eval 2 --train_datasets 'snli' --test_datasets 'snli' --output_dir 'checkpoint3'

# python train.py --batch_size 2 --batch_size_eval 2 --train_datasets 'snli' --test_datasets 'snli' --output_dir 'checkpoint4' --init_checkpoint 'checkpoint1test/model_2.pt'

python train.py --batch_size 2 --batch_size_eval 2 --output_dir 'checkpoint' --init_checkpoint 'checkpoint/model_4.pt'