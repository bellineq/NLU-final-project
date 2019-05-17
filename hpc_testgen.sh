#!/bin/sh
#
#SBATCH --verbose
#SBATCH --job-name=nlu
#SBATCH --output=./out/job_testgen_%j.out
#SBATCH --error=./out/job_testgen_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --mem=6GB
#SBATCH --gres=gpu:1
 
/bin/hostname
/bin/pwd
 

module purge
module load python3/intel/3.6.3
source ~/pyenv/py3.6.3/bin/activate

python testgen.py --batch_size 2 --batch_size_eval 2 --data_dir 'generated_data' --train_datasets 'snli' --test_datasets 'snli_test_c1n2e,snli_test_c2n1e,snli_train_c1n2e,snli_train_c2n1e' --output_dir 'testout/snli' --init_checkpoint 'testmodel/baseline/model.pt'
# python testMNLI.py --batch_size 2 --batch_size_eval 2 --data_dir 'generated_data' --test_datasets 'mnli_train_c1n2,mnli_train_c2n1e,mnli_train_en2n1e,mnli_train_en21c' --output_dir 'testout/mnli_o' --init_checkpoint 'testmodel/mnli_train_1/model_4.pt'
# python testMNLI.py --batch_size 2 --batch_size_eval 2 --data_dir 'data/mt_dnn' --test_datasets 'mnli_matched,mnli_mismatched' --output_dir 'testout/mnli_baseline_train0' --init_checkpoint 'testmodel/mnli_train_0/model_2.pt'
