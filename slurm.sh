#!/bin/bash
#SBATCH --job-name=ttc_train          # Job name
#SBATCH --output=logs/ttc_train_%j.log     # Standard output log
#SBATCH --error=ttc_train_%j.err      # Standard error log
#SBATCH --mail-type=BEGIN,END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=khek.do@ufl.edu # Where to send mail
#SBATCH --nodes=1                     # Run all tasks on a single node
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=4             # Number of CPU cores per task
#SBATCH --mem=32gb                    # Job memory request
#SBATCH --gres=gpu:1                  # Request 1 GPU (e.g. A100, RTX3090, etc.)
#SBATCH --time=08:00:00               # Time limit hrs:min:sec

echo "Job Start"
date;hostname;pwd
echo "---"

module purge
module load python/3.11
# python -m venv .venv

source .venv/bin/activate

# pip install --upgrade pip
# pip install "numpy<2" pandas numexpr torch torchvision matplotlib

python train.py \
    --train-dir /blue/iruchkin/khek.do/sorted_ttc_padding_fix/train \
    --test-dir /blue/iruchkin/khek.do/sorted_ttc_padding_fix/test \
    --backbone custom \
    --epochs 10 \
    --hidden-dim 128 \
    --lstm-layers 1 \
    --seq-len 20 \
    --pred-horizon 10 \
    --lr 1e-4 \
    --dropout 0.3 \
    --weight-decay 1e-3 \
    --restart \
    --save-path best_model.pth

echo "Running post-training evaluation..."
python evaluate.py \
    --test-dir /blue/iruchkin/khek.do/sorted_ttc_padding_fix/test \
    --backbone custom \
    --hidden-dim 128 \
    --lstm-layers 1 \
    --seq-len 20 \
    --pred-horizon 10

echo "Job End"
date
echo "---"