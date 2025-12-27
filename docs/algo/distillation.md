# On-Policy Distillation

On-policy distillation enables training a student model to match a teacher model's behavior by minimizing KL divergence between their output distributions.

## Overview

In on-policy distillation:
- The **student model** generates rollouts (sequences) from prompts
- The **teacher model** computes log probabilities for those sequences  
- Training signal comes from **KL divergence** between teacher and student distributions
- Rewards are set to zero, making KL the sole training objective

This approach allows smaller models to learn from larger, more capable models while maintaining the exploration benefits of on-policy learning.

## Configuration

verl provides dedicated distillation configs that handle all necessary settings:

- `distillation_trainer.yaml` - For FSDP backend
- `distillation_megatron_trainer.yaml` - For Megatron backend

### Basic Usage

```bash
python3 -m verl.trainer.main_ppo \
    --config-name=distillation_trainer \
    actor_rollout_ref.model.path=/path/to/student \
    actor_rollout_ref.ref.model.path=/path/to/teacher \
    data.train_files=/path/to/train.parquet
```

### Example with GSM8K

```bash
# Prepare dataset
python3 examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k

# Run distillation
python3 -m verl.trainer.main_ppo \
    --config-name=distillation_trainer \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    data.train_files=~/data/gsm8k/train.parquet \
    data.val_files=~/data/gsm8k/test.parquet
```

## Key Hyperparameters

### KL Loss Configuration

- **`kl_loss_coef`**: Coefficient for KL loss (default: 1.0)
  - Higher values: student matches teacher more closely
  - Lower values: student has more freedom to diverge
  - For GSM8K, start with 1.0 and adjust based on results

- **`kl_loss_type`**: KL divergence approximation method
  - `"low_var_kl"` (k3): Recommended, better approximation
  - `"kl"` (k1): Simple approximation (faster, less accurate)
  - `"mse"` (k2): Squared difference approximation

- **`entropy_coeff`**: Entropy regularization (default: 0.0)
  - Set to positive value (e.g., 0.01) to encourage exploration
  - Helps prevent mode collapse when student is much smaller than teacher

### Training Hyperparameters

- **`actor.optim.lr`**: Learning rate (default: 1e-5)
  - May need adjustment based on student model size
  - Smaller models often benefit from higher learning rates

- **`actor.ppo_epochs`**: PPO epochs per iteration (default: 1)
  - Can increase for more stable learning
  - Trade-off with computational cost

- **`rollout.temperature`**: Sampling temperature (default: 0.6)
  - Lower: more deterministic, closer to teacher's modes
  - Higher: more exploration, potentially better coverage

## Implementation Details

The distillation leverages verl's existing PPO infrastructure:
- Uses the reference model slot for the teacher model
- Reuses KL divergence computation from PPO's KL penalty
- Efficiently skips unnecessary computations when rewards are zero
- Compatible with all KL divergence approximation methods

### Architecture

1. **Loss Function**: Dedicated `distillation_loss` registered in loss registry
2. **Reward Manager**: `DistillationRewardManager` returns zero rewards
3. **Configuration**: Dedicated trainer configs with proper defaults
4. **Metrics**: Tracks KL divergence, entropy, and distillation-specific metrics

## Monitoring and Evaluation

### Key Metrics

- **`actor/kl_loss`**: KL divergence between student and teacher
  - Should decrease over time
  - Very low values may indicate overfitting

- **`actor/distillation_mode`**: Should always be 1.0
  - Confirms distillation mode is active

- **`actor/entropy_loss`**: Entropy of student's output distribution
  - Monitor to ensure diversity is maintained

### Validation

Use generation metrics to evaluate student quality:
```bash
python3 -m verl.trainer.main_generation \
    --config-name=generation \
    actor_rollout_ref.model.path=/path/to/student/checkpoint \
    data.val_files=/path/to/test.parquet
```

## Troubleshooting

### Student not improving
- Increase `kl_loss_coef` 
- Decrease learning rate
- Check teacher model is loading correctly

### Out of memory
- Reduce `ppo_mini_batch_size`
- Reduce `rollout.response_length`
- Enable gradient checkpointing

### KL divergence exploding
- Decrease `kl_loss_coef`
- Use gradient clipping
- Check for numerical instabilities

## References

- [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/abs/2306.13649)
- ["On-Policy Distillation" from Thinking Machines Lab ](https://thinkingmachines.ai/blog/on-policy-distillation/)