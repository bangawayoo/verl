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
Reward Manager for on-policy distillation in the experimental reward loop.
Returns zero rewards since distillation uses KL divergence loss only.
"""

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase


@register("distillation")
class DistillationRewardManager(RewardManagerBase):
    """
    Reward manager that returns zero rewards for distillation mode.

    In on-policy distillation, we only use KL divergence loss as the training signal.
    By setting rewards to zero, advantages become zero, and policy gradient loss becomes zero,
    leaving only the KL divergence loss between teacher and student models.
    """

    def __init__(self, config, tokenizer, compute_score=None, reward_router_address=None, reward_model_tokenizer=None):
        """
        Initialize the DistillationRewardManager.

        Args:
            config: YAML config (DictConfig)
            tokenizer: Tokenizer for tokenizing messages
            compute_score: Score function (ignored for distillation)
            reward_router_address: Address for reward router (ignored for distillation)
            reward_model_tokenizer: Tokenizer for reward model (ignored for distillation)
        """
        super().__init__(config, tokenizer)
        # These are ignored for distillation but kept for API compatibility
        self.compute_score = compute_score
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer

    async def run_single(self, data: DataProto) -> dict:
        """
        Return zero reward for distillation.

        Args:
            data: DataProto containing single data item

        Returns:
            dict with "reward_score" set to 0.0 and empty "reward_extra_info"
        """
        return {
            "reward_score": 0.0,
            "reward_extra_info": {"distillation_mode": True},
        }

