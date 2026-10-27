"""Episode-local box milestones with per-environment reset and persistence checks."""

import torch


class BoxProgress:
    def __init__(self, num_envs, device, stable_steps=15, grasp_steps=2):
        self.stable_steps = stable_steps
        self.grasp_steps = grasp_steps
        self.grasped = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.lifted = torch.zeros_like(self.grasped)
        self.placed = torch.zeros_like(self.grasped)
        self.success = torch.zeros_like(self.grasped)
        self.stable_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.grasp_count = torch.zeros_like(self.stable_count)

    def reset(self, ids=None):
        ids = slice(None) if ids is None else ids
        for value in (self.grasped, self.lifted, self.placed, self.success, self.stable_count, self.grasp_count):
            value[ids] = 0

    def update(self, holding, cleared, placement):
        self.grasp_count.copy_(torch.where(holding, self.grasp_count + 1, 0))
        self.grasped |= self.grasp_count >= self.grasp_steps
        self.lifted |= self.grasped & cleared
        eligible = self.lifted & placement
        self.placed |= eligible
        self.stable_count.copy_(torch.where(eligible, self.stable_count + 1, 0))
        self.success |= self.stable_count >= self.stable_steps
        return self.success.clone()
