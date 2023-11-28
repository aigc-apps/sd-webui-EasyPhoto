"""Borrowed from https://github.com/kvablack/ddpo-pytorch/blob/main/ddpo_pytorch/stat_tracking.py.
"""

from collections import deque
from typing import List

import numpy as np


class PerPromptStatTracker:
    """Track the mean and std of reward on a per-prompt basis and use that to compute advantages.

    Args:
        buffer_size (int): The number of reward values to store in the buffer for each prompt.
        The buffer persists across epochs.
        min_count (int): The minimum number of reward values to store in the buffer before using the
        per-prompt mean and std. If the buffer contains fewer than `min_count` values, the mean and
        std of the entire batch will be used instead.
    """

    def __init__(self, buffer_size: int, min_count: int):
        self.buffer_size = buffer_size
        self.min_count = min_count
        self.stats = {}

    def update(self, prompts: List[str], rewards: List[float]):
        prompts = np.array(prompts)
        rewards = np.array(rewards)
        unique = np.unique(prompts)
        advantages = np.empty_like(rewards)
        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt]
            if prompt not in self.stats:
                self.stats[prompt] = deque(maxlen=self.buffer_size)
            self.stats[prompt].extend(prompt_rewards)

            if len(self.stats[prompt]) < self.min_count:
                mean = np.mean(rewards)
                std = np.std(rewards) + 1e-6
            else:
                mean = np.mean(self.stats[prompt])
                std = np.std(self.stats[prompt]) + 1e-6
            advantages[prompts == prompt] = (prompt_rewards - mean) / std

        return advantages

    def get_stats(self):
        return {k: {"mean": np.mean(v), "std": np.std(v), "count": len(v)} for k, v in self.stats.items()}
