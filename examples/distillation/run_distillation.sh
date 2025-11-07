set -e

# Configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # Adjust to your GPU setup
PYTHONUNBUFFERED=1

# GSM8K dataset paths
# These will override the paths in gsm8k_distillation_config.yaml
TRAIN_PATH=${HOME}/data/v7/ko/train.parquet
TEST_PATH=${HOME}/data/v7/ko/validation.parquet

# Model paths - UPDATE THESE TO YOUR MODELS
STUDENT_MODEL="/home/ubuntu/kiyoon/checkpoints/gguf/portable/v7/ko-1103/kanana-1.5-2.1b-instruct-2505/checkpoint-90000"  
TEACHER_MODEL="/home/ubuntu/kiyoon/checkpoints/gguf/portable/v7/ko-1103/kanana-1.5-8b-instruct-2505/checkpoint-150000"


EXPERIMENT_NAME="kanana-ko-v7-lr1e-6"

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

# Run distillation training using the distillation_config.yaml
# The YAML file contains all distillation-specific settings (KL loss, reward manager, etc.)
# We only override the data paths and model paths here since they use shell variables
cd "$VERL_ROOT"
python3 -m verl.trainer.main_ppo \
    --config-name=distillation_config \
    --config-path="$SCRIPT_DIR" \
    data.train_files="$TRAIN_PATH" \
    data.val_files="$TEST_PATH" \
    actor_rollout_ref.model.path="$STUDENT_MODEL" \
    actor_rollout_ref.ref.model.path="$TEACHER_MODEL" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    "$@" 2>&1 | tee "$VERL_ROOT/verl_distillation.log"

echo "=========================================="
echo "Training complete! Check log for details:"
echo "$VERL_ROOT/verl_distillation.log"
echo "=========================================="

