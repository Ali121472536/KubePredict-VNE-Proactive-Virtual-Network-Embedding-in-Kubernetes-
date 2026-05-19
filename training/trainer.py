# training/trainer.py

import numpy as np

import torch
import torch.nn.functional as F

from models.gat_encoder import substrate_to_pyg


class PPOTrainer:

    # ====================================================
    # INITIALIZATION
    # ====================================================

    def __init__(self,
                 env,
                 gat_model,
                 ppo_agent,
                 transformer_model,
                 optimizer,
                 device):

        self.env = env

        self.gat_model = gat_model

        self.ppo_agent = ppo_agent

        self.transformer_model = transformer_model

        self.optimizer = optimizer

        self.device = device

        self.gamma = 0.99

        self.eps_clip = 0.2

    # ====================================================
    # GAT GRAPH EMBEDDING
    # ====================================================

    def get_graph_embedding(self):

        pyg_graph = substrate_to_pyg(
            self.env.substrate
        ).to(self.device)

        node_embeddings = self.gat_model(
            pyg_graph.x,
            pyg_graph.edge_index
        )

        graph_embedding = torch.mean(
            node_embeddings,
            dim=0
        )

        return graph_embedding.unsqueeze(0)

    # ====================================================
    # TEMPORAL WORKLOAD GENERATION
    # ====================================================

    def generate_workload_sequence(self):

        # ==============================================
        # BASE DEMAND
        # ==============================================

        base_load = np.random.uniform(
            10,
            20
        )

        # ==============================================
        # TEMPORAL TREND
        # ==============================================

        trend = np.random.uniform(
            1,
            3
        )

        history = []

        # ==============================================
        # GENERATE TEMPORAL SERIES
        # ==============================================

        for t in range(5):

            noise = np.random.normal(
                0,
                1
            )

            demand = (

                base_load

                + trend * t

                + noise
            )

            demand = max(
                demand,
                1
            )

            history.append(demand)

        # ==============================================
        # INPUT SEQUENCE
        # ==============================================

        input_sequence = history[:-1]

        # ==============================================
        # TARGET DEMAND
        # ==============================================

        target_value = history[-1]

        # ==============================================
        # CONVERT TO TENSORS
        # ==============================================

        sequence_tensor = torch.tensor(
            [[[x] for x in input_sequence]],
            dtype=torch.float
        ).to(self.device)

        target_tensor = torch.tensor(
            [[target_value]],
            dtype=torch.float
        ).to(self.device)

        return sequence_tensor, target_tensor

    # ====================================================
    # TRAIN ONE PPO EPISODE
    # ====================================================

    def train_episode(self):

        # ==============================================
        # RESET ENVIRONMENT
        # ==============================================

        state = self.env.reset()

        done = False

        total_reward = 0

        total_loss = 0

        step_count = 0

        # ==============================================
        # SEQUENTIAL ORCHESTRATION LOOP
        # ==============================================

        while not done:

            # ==========================================
            # GAT EMBEDDING
            # ==========================================

            graph_embedding = \
                self.get_graph_embedding()

            # ==========================================
            # WORKLOAD HISTORY
            # ==========================================

            workload_sequence, target_demand = \
                self.generate_workload_sequence()

            # ==========================================
            # TRANSFORMER FORECAST
            # ==========================================

            predicted_tensor, confidence = \
                self.transformer_model(
                    workload_sequence
                )

            predicted_demand = \
                predicted_tensor.item()

            confidence_score = \
                confidence.item()

            # ==========================================
            # CONFIDENCE-WEIGHTED FORECAST
            # ==========================================

            predicted_demand = (
                predicted_demand
                * confidence_score
            )

            # ==========================================
            # FORECAST LOSS
            # ==========================================

            forecast_loss = F.mse_loss(
                predicted_tensor,
                target_demand
            )

            # ==========================================
            # PPO ACTION SELECTION
            # ==========================================

            action, old_log_prob, value = \
                self.ppo_agent.select_action(
                    graph_embedding
                )

            # ==========================================
            # ENVIRONMENT STEP
            # ==========================================

            next_state, reward, done = \
                self.env.step(
                    action,
                    predicted_demand
                )

            # ==========================================
            # REWARD ACCUMULATION
            # ==========================================

            total_reward += reward

            # ==========================================
            # NORMALIZE REWARD
            # ==========================================

            normalized_reward = reward / 1000.0

            reward_tensor = torch.tensor(
                [normalized_reward],
                dtype=torch.float
            ).to(self.device)

            # ==========================================
            # PPO EVALUATION
            # ==========================================

            log_prob, state_value, entropy = \
                self.ppo_agent.evaluate(
                    graph_embedding,
                    torch.tensor(
                        [action]
                    ).to(self.device)
                )

            # ==========================================
            # ADVANTAGE ESTIMATION
            # ==========================================

            advantage = (

                reward_tensor.view(-1)

                -

                state_value.view(-1)
            )

            # ==========================================
            # PPO RATIO
            # ==========================================

            ratio = torch.exp(

                log_prob

                -

                old_log_prob.detach()
            )

            # ==========================================
            # PPO OBJECTIVE
            # ==========================================

            surr1 = ratio * advantage

            surr2 = torch.clamp(
                ratio,
                1 - self.eps_clip,
                1 + self.eps_clip
            ) * advantage

            actor_loss = -torch.min(
                surr1,
                surr2
            ).mean()

            # ==========================================
            # CRITIC LOSS
            # ==========================================

            critic_loss = F.mse_loss(
                state_value.view(-1),
                reward_tensor.view(-1)
            )

            # ==========================================
            # ENTROPY BONUS
            # ==========================================

            entropy_loss = -0.01 * entropy.mean()

            # ==========================================
            # TOTAL JOINT LOSS
            # ==========================================

            loss = (

                actor_loss

                + 0.5 * critic_loss

                + entropy_loss

                + 0.1 * forecast_loss
            )

            # ==========================================
            # BACKPROPAGATION
            # ==========================================

            self.optimizer.zero_grad()

            loss.backward()

            # ==========================================
            # GRADIENT CLIPPING
            # ==========================================

            torch.nn.utils.clip_grad_norm_(
                self.gat_model.parameters(),
                max_norm=1.0
            )

            torch.nn.utils.clip_grad_norm_(
                self.ppo_agent.parameters(),
                max_norm=1.0
            )

            torch.nn.utils.clip_grad_norm_(
                self.transformer_model.parameters(),
                max_norm=1.0
            )

            self.optimizer.step()

            # ==========================================
            # UPDATE STATE
            # ==========================================

            state = next_state

            total_loss += loss.item()

            step_count += 1

        # ==============================================
        # EPISODE AVERAGES
        # ==============================================

        avg_loss = total_loss / max(
            step_count,
            1
        )

        acceptance_ratio = \
            self.env.get_acceptance_ratio()

        return (
            total_reward,
            avg_loss,
            acceptance_ratio
        )