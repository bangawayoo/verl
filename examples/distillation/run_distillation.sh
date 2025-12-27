set -e

# Configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # Adjust to your GPU setup
export PYTHONUNBUFFERED=1

TRAIN_PATH=${HOME}/data/v7/ko/train.parquet
TEST_PATH=${HOME}/data/v7/ko/validation.parquet
# TRAIN_PATH=${HOME}/data/jsonl_processed/train.parquet
# TEST_PATH=${HOME}/data/jsonl_processed/validation.parquet

# Model paths - UPDATE THESE TO YOUR MODELS
STUDENT_MODEL="/home/ubuntu/kiyoon/checkpoints/gguf/portable/v7/ko-1103/kanana-1.5-2.1b-instruct-2505/checkpoint-90000"  
TEACHER_MODEL="/home/ubuntu/kiyoon/checkpoints/gguf/portable/v7/ko-1103/kanana-1.5-8b-instruct-2505/checkpoint-150000"


LR=1e-6
EXPERIMENT_NAME="kanana-ko-v7-lr1e-6-n=2"


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

