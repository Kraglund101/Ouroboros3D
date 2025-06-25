#!/bin/bash
#SBATCH --job-name=svd_train_infer
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH -p gpu

# ✅ Create logs directory if it doesn't exist
mkdir -p logs
# ✅ Load modules first

module load cuda/12.5
module load gcc/8.5.0
module load python/3.9.9

# ✅ Activate virtual environment
echo "🔄 Activating virtual environment..."
source ~/Ouroboros3D/venv/bin/activate
echo "✅ Activated: $VIRTUAL_ENV"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo ""

# ✅ Check torchrun is available
which torchrun || echo "❌ torchrun not found!"

# ✅ Run training jobs
torchrun --nnodes=1 --nproc_per_node=1 train.py --config configs/mv/svd_lgm_rgb_ccm_lvis_independent.yaml
torchrun --nnodes=1 --nproc_per_node=1 train.py --config configs/mv/svd_lgm_rgb_ccm_lvis_interpolated.yaml

echo "✅ Finished training script"