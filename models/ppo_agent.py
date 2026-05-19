# models/ppo_agent.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical


class PPOAgent(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 action_dim):

        super(PPOAgent, self).__init__()

        # =========================
        # ACTOR
        # =========================

        self.actor = nn.Sequential(

            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, action_dim)
        )

        # =========================
        # CRITIC
        # =========================

        self.critic = nn.Sequential(

            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, 1)
        )

    # ==========================================
    # SELECT ACTION
    # ==========================================

    def select_action(self, state):

        logits = self.actor(state)

        probs = F.softmax(logits, dim=-1)

        dist = Categorical(probs)

        action = dist.sample()

        log_prob = dist.log_prob(action)

        value = self.critic(state)

        return action.item(), log_prob, value

    # ==========================================
    # EVALUATE ACTION
    # ==========================================

    def evaluate(
            self,
            states,
            actions
    ):
        # ==========================================
        # ACTOR
        # ==========================================

        action_logits = self.actor(states)

        # ==========================================
        # NUMERICAL STABILITY
        # ==========================================

        action_logits = torch.clamp(
            action_logits,
            -20,
            20
        )

        action_probs = F.softmax(
            action_logits,
            dim=-1
        )

        action_probs = torch.nan_to_num(
            action_probs,
            nan=1e-8
        )

        action_probs = action_probs / (
                action_probs.sum(
                    dim=-1,
                    keepdim=True
                ) + 1e-8
        )

        dist = Categorical(
            action_probs
        )

        action_log_probs = dist.log_prob(
            actions
        )

        entropy = dist.entropy()

        # ==========================================
        # CRITIC
        # ==========================================

        state_value = self.critic(states)

        return (
            action_log_probs,
            state_value,
            entropy
        )

        # ==========================================
        # REMOVE NaNs
        # ==========================================

        action_probs = torch.nan_to_num(
            action_probs,
            nan=1e-8
        )

        # ==========================================
        # RENORMALIZE
        # ==========================================

        action_probs = action_probs / (
                action_probs.sum(dim=-1, keepdim=True)
                + 1e-8
        )

        dist = Categorical(probs)

        log_probs = dist.log_prob(actions)

        entropy = dist.entropy()

        values = self.critic(states)

        return log_probs, values, entropy