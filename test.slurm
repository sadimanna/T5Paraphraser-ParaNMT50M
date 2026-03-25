#!/bin/bash

#SBATCH --job-name=paraphraser
#SBATCH --output=slurm_logs/master_%j.out
#SBATCH --error=slurm_logs/master_%j.err
#SBATCH --partition=short
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G

# Load Python module
# module load cuda/12.4
mkdir -p slurm_logs
# -----------------------------------------------------------------------------
# Activate virtual environment (SLURM-safe)
# -----------------------------------------------------------------------------

echo "Activating virtual environment..."

if [[ -z "${SLURM_SUBMIT_DIR:-}" ]]; then
    echo "ERROR: SLURM_SUBMIT_DIR is not set"
    exit 1
fi

echo "SLURM_SUBMIT_DIR = ${SLURM_SUBMIT_DIR}"

# breaching/ is the submit directory
BASE_DIR="${SLURM_SUBMIT_DIR}"

# project root is one level above
PROJECT_ROOT="$(cd "${BASE_DIR}/.." && pwd)"

echo "BASE_DIR     = ${BASE_DIR}"
echo "PROJECT_ROOT = ${PROJECT_ROOT}"

cd "${PROJECT_ROOT}"

if [[ ! -f bin/activate ]]; then
    echo "ERROR: Virtual environment not found at ${PROJECT_ROOT}/bin/activate"
    exit 1
fi

source bin/activate

# Sanity checks
which python
python --version

# Metrics dependencies
# python -m pip install -q torchmetrics lpips

# Return to breaching
cd "${BASE_DIR}"

echo "Virtual environment activated successfully."

NUM_GPUS=${NUM_GPUS:-2}
NUM_NODES=${SLURM_NNODES:-1}
NODE_RANK=${SLURM_NODEID:-0}

export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=12345
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_FR_BUFFER_SIZE=1

srun python test_cached_dataset.py