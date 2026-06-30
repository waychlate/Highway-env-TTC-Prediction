#!/bin/bash
#SBATCH --job-name=ttc_train          # Job name
#SBATCH --output=ttc_train_%j.log     # Standard output log
#SBATCH --error=ttc_train_%j.err      # Standard error log
#SBATCH --mail-type=BEGIN,END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user= # Where to send mail
#SBATCH --nodes=1                     # Run all tasks on a single node
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=4             # Number of CPU cores per task
#SBATCH --mem=32gb                    # Job memory request
#SBATCH --gres=gpu:1                  # Request 1 GPU (e.g. A100, RTX3090, etc.)
#SBATCH --time=04:00:00               # Time limit hrs:min:sec

echo "Job Start"
date;hostname;pwd
echo "---"

module purge
module load python/3.11
module load pytorch/2.0.1

python train.py \
    --train-dir /x/x/x \
    --test-dir /x/x/x  \
    --epochs 10 \
    --batch-size 32 \
    --seq-len 10 \
    --lr 3e-5 \
    --dropout 0.2 \
    --unfreeze-backbone \
    --loss-fn huber \
    --use-scheduler \
    --restart \
    --save-path best_model.pth

echo "Running post-training evaluation..."
python evaluate.py \
    --test-dir /foo/bar \
    --model-path best_model.pth \
    --seq-len 10 \
    --output-plot ttc_predictions_comparison.png

echo "Job End"
date
echo "---"