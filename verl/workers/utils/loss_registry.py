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
Registry for loss functions used in training.

This module provides a registry pattern for managing different loss functions
(PPO, SFT, distillation, etc.) and allows easy extension with new loss types.
"""

from typing import Dict, Callable

_LOSS_FUNCTIONS: Dict[str, Callable] = {}


def register_loss(name: str):
    """
    Decorator to register a loss function.
    
    Args:
        name: The name to register the loss function under.
        
    Returns:
        The decorator function.
        
    Example:
        @register_loss("ppo")
        def ppo_loss(config, model_output, data, dp_group=None):
            ...
    """
    def decorator(func):
        if name in _LOSS_FUNCTIONS:
            raise ValueError(f"Loss function '{name}' is already registered")
        _LOSS_FUNCTIONS[name] = func
        return func
    return decorator


def get_loss_function(name: str) -> Callable:
    """
    Retrieve a registered loss function by name.
    
    Args:
        name: The name of the loss function to retrieve.
        
    Returns:
        The loss function.
        
    Raises:
        ValueError: If the loss function name is not registered.
    """
    if name not in _LOSS_FUNCTIONS:
        available = ", ".join(sorted(_LOSS_FUNCTIONS.keys()))
        raise ValueError(f"Unknown loss function: '{name}'. Available: {available}")
    return _LOSS_FUNCTIONS[name]


def list_loss_functions() -> list[str]:
    """
    List all registered loss function names.
    
    Returns:
        A sorted list of registered loss function names.
    """
    return sorted(_LOSS_FUNCTIONS.keys())
