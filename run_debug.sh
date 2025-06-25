#!/bin/bash
#SBATCH --job-name=hold-interactive
#SBATCH --output=logs/hold-%j.out
#SBATCH --error=logs/hold-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=frederik.kraglund@gmail.com

echo "🔒 Holding H100 node with sleep infinity..."
echo "💡 Use 'srun --jobid=\$SLURM_JOB_ID --pty bash' to access your shell."
echo "💡 Inside the shell, use tmux or run scripts as usual."
echo ""

# Keep the job alive indefinitely so you can attach
sleep infinity
