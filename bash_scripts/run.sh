#!/bin/bash -l
#SBATCH --gres=gpu:1
#SBATCH --time=23:00:00
#SBATCH --partition=a100
#SBATCH -o /home/hpc/v101be/v101be15/finetuning_text_classification/slurm_logs/%A-%a.%x.o
#SBATCH -e /home/hpc/v101be/v101be15/finetuning_text_classification/slurm_logs/%A-%a.%x.e
#SBATCH --gres=gpu:1
#SBATCH --array 1-750%20


export PYTHONPATH="${PYTHONPATH}:${HOME}/finetuning_text_classification/"

args_file="$1"
echo $args_file
ARGS_PATH=/home/hpc/v101be/v101be15/finetuning_text_classification/bash_args/${args_file}.args
ARGS=$(sed "${SLURM_ARRAY_TASK_ID}q;d" $ARGS_PATH)
source ~/miniconda3/bin/activate finetuning_text_classifiers
python finetune.py $ARGS