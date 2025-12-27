set -e

# Configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # Adjust to your GPU setup
export PYTHONUNBUFFERED=1

# Dataset paths - GSM8K (open source math dataset)
# Run this first to prepare the data:
#   python3 examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k
TRAIN_PATH=${HOME}/data/gsm8k/train.parquet
TEST_PATH=${HOME}/data/gsm8k/test.parquet

# Model paths - Open source Qwen models from Hugging Face
# Option 1: True distillation (large teacher -> small student)
STUDENT_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
TEACHER_MODEL="Qwen/Qwen2.5-7B-Instruct"

# Option 2: Self-distillation (same model, uncomment to use)
# STUDENT_MODEL="Qwen/Qwen2.5-3B-Instruct"
# TEACHER_MODEL="Qwen/Qwen2.5-3B-Instruct"

# Training hyperparameters
LR=1e-6
EXPERIMENT_NAME="gsm8k-distillation-qwen-0.5b-from-7b"

echo "=========================================="
echo "On-Policy Distillation Training"
echo "=========================================="
echo "Student Model: $STUDENT_MODEL"
echo "Teacher Model: $TEACHER_MODEL"
echo "Train Data: $TRAIN_PATH"
echo "Test Data: $TEST_PATH"
echo "=========================================="

# Get the script directory and verl root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Run distillation training using the dedicated distillation_trainer config
# This config properly sets up all distillation-specific settings
# We only override the data paths and model paths here since they use shell variables
cd "$VERL_ROOT"
python3 -m verl.trainer.main_ppo \
    --config-name=distillation_trainer \
    data.train_files="$TRAIN_PATH" \
    data.val_files="$TEST_PATH" \
    actor_rollout_ref.model.path="$STUDENT_MODEL" \
    actor_rollout_ref.ref.model.path="$TEACHER_MODEL" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    actor_rollout_ref.actor.optim.lr="$LR" \
    actor_rollout_ref.rollout.n=2 \
    data.train_batch_size=256 \
    trainer.total_epochs=1 \
    "$@" 2>&1 | tee "$VERL_ROOT/verl_distillation.log"

echo "=========================================="
echo "Training complete! Check log for details:"
echo "$VERL_ROOT/verl_distillation.log"
echo "=========================================="
