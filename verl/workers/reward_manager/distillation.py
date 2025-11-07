# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Reward Manager for on-policy distillation.
Returns zero token-level rewards matching the response_mask shape.
"""

from typing import Any

import torch

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("distillation")
class DistillationRewardManager(AbstractRewardManager):
    """
    Reward manager that returns zero rewards for distillation mode.
    
    In on-policy distillation, we only use KL divergence loss as the training signal.
    By setting rewards to zero, advantages become zero, and policy gradient loss becomes zero,
    leaving only the KL divergence loss between teacher and student models.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", **kwargs) -> None:
        """
        Initialize the DistillationRewardManager.
        
        Args:
            tokenizer: Tokenizer (ignored, kept for compatibility)
            num_examine: Number of examples to print (ignored, kept for compatibility)
            compute_score: Score function (ignored, kept for compatibility)
            reward_fn_key: Key for data source (ignored, kept for compatibility)
            **kwargs: Additional kwargs (ignored)
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """
        Return zero rewards matching the response mask shape.
        
        Args:
            data: DataProto containing batch data
            return_dict: If True, return dictionary with reward_tensor and reward_extra_info
            
        Returns:
            If return_dict=False: torch.Tensor of shape (batch_size, response_length) filled with zeros
            If return_dict=True: dict with "reward_tensor" key containing zero tensor
        """
        # Get response mask to determine shape
        if "response_mask" in data.batch:
            response_mask = data.batch["response_mask"]
            reward_tensor = torch.zeros_like(response_mask, dtype=torch.float32)
        elif "responses" in data.batch:
            # Fallback: use responses shape
            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        else:
            # Last fallback: infer from batch size
            batch_size = data.batch_size[0] if hasattr(data, "batch_size") else len(data)
            reward_tensor = torch.zeros((batch_size, 1), dtype=torch.float32)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": {},
            }
        else:
            return reward_tensor


