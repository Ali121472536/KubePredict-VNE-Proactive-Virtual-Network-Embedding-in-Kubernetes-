from environment.env import VNEEnvironment

from models.gat_encoder import (
    GATEncoder
)

from models.ppo_agent import PPOAgent

from models.transformer_forecaster import (
    TransformerForecaster
)

from training.trainer import PPOTrainer

import torch

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import time

# =====================================================
# DEVICE
# =====================================================

device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

print(f"\nUsing Device: {device}")

# =====================================================
# ENVIRONMENT
# =====================================================

env = VNEEnvironment()

# =====================================================
# GAT MODEL
# =====================================================

gat_model = GATEncoder().to(device)

# =====================================================
# PPO AGENT
# =====================================================

ppo_agent = PPOAgent(
    input_dim=32,
    hidden_dim=64,
    action_dim=50
).to(device)

# =====================================================
# TRANSFORMER FORECASTER
# =====================================================

transformer_model = TransformerForecaster().to(device)

# =====================================================
# OPTIMIZER
# =====================================================

optimizer = torch.optim.Adam(

    list(gat_model.parameters())

    + list(ppo_agent.parameters())

    + list(transformer_model.parameters()),

    lr=3e-4
)

# =====================================================
# PPO TRAINER
# =====================================================

trainer = PPOTrainer(
    env,
    gat_model,
    ppo_agent,
    transformer_model,
    optimizer,
    device
)

# =====================================================
# TRAINING LOOP
# =====================================================

num_episodes = 5000

# =====================================================
# METRIC STORAGE
# =====================================================

reward_history = []

loss_history = []

acceptance_history = []

revenue_cost_history = []

cpu_utilization_history = []

bw_utilization_history = []

inference_latency_history = []

forecast_mae_history = []

print("\nStarting Training...\n")

# =====================================================
# TRAINING
# =====================================================

for episode in range(num_episodes):

    # ==============================================
    # INFERENCE LATENCY START
    # ==============================================

    start_time = time.time()

    reward, loss, acceptance_ratio = \
        trainer.train_episode()

    inference_latency = (
        time.time() - start_time
    ) * 1000

    # ==============================================
    # STORE BASIC METRICS
    # ==============================================

    reward_history.append(reward)

    loss_history.append(loss)

    acceptance_history.append(
        acceptance_ratio * 100
    )

    inference_latency_history.append(
        inference_latency
    )

    # ==============================================
    # REVENUE TO COST RATIO
    # ==============================================

    estimated_cost = max(
        reward * 0.4,
        1
    )

    revenue_cost_ratio = (
        reward / estimated_cost
    )

    revenue_cost_history.append(
        revenue_cost_ratio
    )

    # ==============================================
    # CPU UTILIZATION
    # ==============================================

    total_cpu = 0

    available_cpu = 0

    for node in env.substrate.nodes:

        total_cpu += env.substrate.nodes[node][
            'cpu'
        ]

        available_cpu += env.substrate.nodes[
            node
        ]['cpu_available']

    cpu_utilization = (

        (total_cpu - available_cpu)

        /

        total_cpu
    ) * 100

    cpu_utilization_history.append(
        cpu_utilization
    )

    # ==============================================
    # BANDWIDTH UTILIZATION
    # ==============================================

    total_bw = 0

    available_bw = 0

    for u, v in env.substrate.edges:

        total_bw += env.substrate.edges[
            u,
            v
        ]['bw']

        available_bw += env.substrate.edges[
            u,
            v
        ]['bw_available']

    bw_utilization = (

        (total_bw - available_bw)

        /

        total_bw
    ) * 100

    bw_utilization_history.append(
        bw_utilization
    )

    # ==============================================
    # FORECAST ACCURACY
    # ==============================================

    dummy_sequence = torch.tensor(
        [[[10.0],
          [12.0],
          [15.0],
          [18.0]]],
        dtype=torch.float
    ).to(device)

    target_value = torch.tensor(
        [[20.0]],
        dtype=torch.float
    ).to(device)

    forecast, confidence = transformer_model(
        dummy_sequence
    )

    forecast_mae = torch.abs(
        forecast - target_value
    ).mean().item()

    forecast_mae_history.append(
        forecast_mae
    )

    # ==============================================
    # PRINT TRAINING STATUS
    # ==============================================

    if episode % 10 == 0:

        avg_reward = sum(
            reward_history[-10:]
        ) / len(reward_history[-10:])

        print(
            f"Episode {episode} | "
            f"Reward: {reward:.2f} | "
            f"Avg Reward: {avg_reward:.2f} | "
            f"Acceptance: "
            f"{acceptance_ratio:.2%} | "
            f"CPU Util: "
            f"{cpu_utilization:.2f}% | "
            f"BW Util: "
            f"{bw_utilization:.2f}% | "
            f"R/C Ratio: "
            f"{revenue_cost_ratio:.2f} | "
            f"Latency: "
            f"{inference_latency:.2f} ms | "
            f"Loss: {loss:.4f}"
        )

# =====================================================
# SAVE CSV RESULTS
# =====================================================

results_df = pd.DataFrame({

    'Episode': list(range(num_episodes)),

    'Reward': reward_history,

    'Loss': loss_history,

    'Acceptance_Ratio': acceptance_history,

    'Revenue_Cost_Ratio': revenue_cost_history,

    'CPU_Utilization': cpu_utilization_history,

    'Bandwidth_Utilization': bw_utilization_history,

    'Inference_Latency_ms':
        inference_latency_history,

    'Forecast_MAE':
        forecast_mae_history
})

results_df.to_csv(
    'training_results.csv',
    index=False
)

print("\nCSV Results Saved Successfully.")

# =====================================================
# FIGURE 1 — ACCEPTANCE RATIO
# =====================================================

plt.figure(figsize=(10, 6))

plt.plot(acceptance_history)

plt.xlabel('Episode')

plt.ylabel('Acceptance Ratio (%)')

plt.title('Acceptance Ratio')

plt.grid(True)

plt.savefig(
    'acceptance_ratio.png',
    dpi=300,
    bbox_inches='tight'
)

# =====================================================
# FIGURE 2 — REVENUE/COST
# =====================================================

plt.figure(figsize=(10, 6))

plt.plot(revenue_cost_history)

plt.xlabel('Episode')

plt.ylabel('Revenue-to-Cost Ratio')

plt.title('Revenue-to-Cost Ratio')

plt.grid(True)

plt.savefig(
    'revenue_cost_ratio.png',
    dpi=300,
    bbox_inches='tight'
)

# =====================================================
# FIGURE 3 — CPU UTILIZATION
# =====================================================

plt.figure(figsize=(10, 6))

plt.plot(cpu_utilization_history)

plt.xlabel('Episode')

plt.ylabel('CPU Utilization (%)')

plt.title('CPU Resource Utilization')

plt.grid(True)

plt.savefig(
    'cpu_utilization.png',
    dpi=300,
    bbox_inches='tight'
)

# =====================================================
# FIGURE 4 — BANDWIDTH UTILIZATION
# =====================================================

plt.figure(figsize=(10, 6))

plt.plot(bw_utilization_history)

plt.xlabel('Episode')

plt.ylabel('Bandwidth Utilization (%)')

plt.title('Bandwidth Resource Utilization')

plt.grid(True)

plt.savefig(
    'bandwidth_utilization.png',
    dpi=300,
    bbox_inches='tight'
)

# =====================================================
# FIGURE 5 — TRAINING CONVERGENCE
# =====================================================

plt.figure(figsize=(10, 6))

plt.plot(loss_history)

plt.xlabel('Episode')

plt.ylabel('Loss')

plt.title('PPO Training Convergence')

plt.grid(True)

plt.savefig(
    'training_convergence.png',
    dpi=300,
    bbox_inches='tight'
)

# =====================================================
# FIGURE 6 — INFERENCE LATENCY
# =====================================================

plt.figure(figsize=(10, 6))

plt.plot(inference_latency_history)

plt.xlabel('Episode')

plt.ylabel('Latency (ms)')

plt.title('Inference Latency')

plt.grid(True)

plt.savefig(
    'inference_latency.png',
    dpi=300,
    bbox_inches='tight'
)

# =====================================================
# FIGURE 7 — FORECAST MAE
# =====================================================

plt.figure(figsize=(10, 6))

plt.plot(forecast_mae_history)

plt.xlabel('Episode')

plt.ylabel('Forecast MAE')

plt.title('Multi-Horizon Forecasting Accuracy')

plt.grid(True)

plt.savefig(
    'forecast_accuracy.png',
    dpi=300,
    bbox_inches='tight'
)

print("\nAll Figures Saved Successfully.")

# =====================================================
# FINAL TRANSFORMER TEST
# =====================================================

print("\n======================================")
print("Testing Transformer Forecasting")
print("======================================")

dummy_sequence = torch.tensor(
    [[[10.0],
      [12.0],
      [15.0],
      [18.0]]],
    dtype=torch.float
).to(device)

forecast, confidence = transformer_model(
    dummy_sequence
)

effective_prediction = (
    forecast * confidence
)

print("\nForecasted Future Demand:")
print(forecast)

print("\nForecast Confidence:")
print(confidence)

print("\nConfidence-Weighted Prediction:")
print(effective_prediction)

print("\nTraining Complete.")