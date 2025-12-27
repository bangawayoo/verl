# On-Policy Distillation Example

This example demonstrates how to use verl for **on-policy distillation**, where a student model learns from a teacher model by minimizing KL divergence between their log probabilities.

## Overview

In on-policy distillation:
- **Student model** generates rollouts (sequences)
- **Teacher model** (reference model) computes log probabilities for those sequences
- **Loss** is KL divergence between teacher and student log probabilities
- **Rewards** are set to zero, so only KL divergence signal is used for training

## Key Components

### 1. Zero Rewards

We use a specialized reward manager (`distillation`) that returns zero rewards. This ensures:
- Advantages become zero
- Policy gradient loss becomes zero
- Only KL divergence loss remains as the training signal

### 2. KL Loss Configuration

Enable KL loss in actor config:
```yaml
actor_rollout_ref:
  actor:
    use_kl_loss: true
    kl_loss_type: low_var_kl  # Recommended: better approximation
    kl_loss_coef: 1.0  # Adjust this hyperparameter
```

### 3. Teacher Model Configuration

Specify teacher model path in reference config:
```yaml
actor_rollout_ref:
  ref:
    model:
      path: /path/to/teacher/model
```

## Configuration Files

verl provides dedicated distillation trainer configurations:
- `verl/trainer/config/distillation_trainer.yaml`: For FSDP backend
- `verl/trainer/config/distillation_megatron_trainer.yaml`: For Megatron backend

Example scripts:
- `run_distillation.sh`: Training script using the distillation trainer config
- `run_evaluation.sh`: Evaluation script for trained models
- `evaluate_model.py`: Python script for flexible model evaluation

## Usage

### Quick Start Example

1. **Prepare your dataset** (example with GSM8K):
```bash
python3 examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k
```

2. **Run distillation training**:
```bash
# From verl root directory
bash examples/distillation/run_distillation.sh

# Or customize the models and data paths:
bash examples/distillation/run_distillation.sh \
    data.train_files=/path/to/train.parquet \
    data.val_files=/path/to/val.parquet \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.ref.model.path=Qwen/Qwen2.5-7B-Instruct
```

**Note**: The script uses environment variables for default paths. Update the script or override via command line.

### General Usage (Custom Dataset)

Use the dedicated distillation trainer config:

```bash
python3 -m verl.trainer.main_ppo \
    --config-name=distillation_trainer \
    actor_rollout_ref.model.path=/path/to/student/model \
    actor_rollout_ref.ref.model.path=/path/to/teacher/model \
    data.train_files=/path/to/train.parquet \
    data.val_files=/path/to/test.parquet
```

The distillation trainer config automatically sets up:
- Distillation loss function (`loss_mode: distillation`)
- Zero rewards via `DistillationRewardManager`
- Disabled critic (not needed for distillation)
- Proper KL loss configuration

## Hyperparameters

### Key Hyperparameters

- **`kl_loss_coef`**: Coefficient for KL loss (default: 1.0)
  - Higher values: student matches teacher more closely
  - Lower values: student has more freedom to diverge

- **`kl_loss_type`**: KL divergence approximation method
  - `"low_var_kl"` (k3): Recommended, better approximation
  - `"kl"` (k1): Simple approximation (faster, less accurate)
  - `"mse"` (k2): Squared difference approximation

- **`entropy_coeff`**: Entropy regularization (default: 0.0)
  - Set to positive value (e.g., 0.01) to encourage exploration

### Monitoring Metrics

Watch these metrics during training:
- `actor/kl_loss`: KL divergence between student and teacher
- `actor/distillation_mode`: Should be 1.0 (indicates zero advantages detected)
- `actor/pg_loss`: Should be ~0.0 (policy gradient is zero)

## How It Works

### Architecture Overview

1. **Rollout Phase**: Student model generates sequences from prompts
2. **Reference Log Prob**: Teacher model (configured as reference) computes log probabilities for student's sequences
3. **Zero Rewards**: `DistillationRewardManager` returns zeros → advantages become zero
4. **Loss Computation**: 
   - Dedicated `distillation_loss` function computes KL divergence
   - No policy gradient computation (cleaner than detecting zero advantages)
   - Optional entropy regularization
5. **Training**: Only KL loss (and optional entropy) contributes to gradient updates

### Implementation Details

The distillation is implemented as a first-class training objective:
- Dedicated `distillation_loss` function in the loss registry
- Clean separation from PPO loss logic
- Uses the reference model slot for the teacher model
- Reuses KL divergence computation from PPO's KL penalty
- Compatible with all KL divergence approximation methods (`kl`, `mse`, `low_var_kl`, etc.)

## Example: Using Custom Reward Function

Alternatively, you can use a custom reward function:

1. Create reward function file (e.g., `my_distillation_reward.py`):
```python
def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    return 0.0
```

2. Configure in YAML:
```yaml
custom_reward_function:
  path: my_distillation_reward
  name: compute_score

reward_model:
  enable: False
```

## Evaluation

After training, evaluate your model's performance:

### Quick Evaluation (Shell Script)
```bash
# Edit run_evaluation.sh to set your checkpoint path, then run:
bash run_evaluation.sh
```

### Flexible Evaluation (Python Script)
```bash
# Evaluate a checkpoint
python3 evaluate_model.py --model_path /path/to/checkpoint/actor

# See all options
python3 evaluate_model.py --help
```

**For detailed evaluation guide**, see [EVALUATION_README.md](./EVALUATION_README.md)

## Troubleshooting

### Advantages are not zero
- Verify `reward_manager: distillation` is set correctly
- Check that reward tensor shape matches response_mask
- Ensure `use_kl_in_reward: false` is set

### KL loss is not decreasing
- Try increasing `kl_loss_coef`
- Check teacher and student models are compatible
- Verify reference model path is correct

### Out of memory
- Reduce `ppo_mini_batch_size`
- Reduce `train_batch_size`
- Use gradient checkpointing if available

## References

- Loss implementation: `verl/workers/roles/utils/losses.py` (see `distillation_loss`)
- Loss registry: `verl/workers/roles/utils/loss_registry.py`
- Reward manager: `verl/workers/reward_manager/distillation.py`
- KL divergence functions: `verl/trainer/ppo/core_algos.py`
- Trainer configs: `verl/trainer/config/distillation_trainer.yaml`
- Algorithm documentation: `docs/algo/distillation.md`

