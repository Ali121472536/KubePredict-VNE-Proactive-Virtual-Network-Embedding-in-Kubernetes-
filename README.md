# KubePredict-VNE: Proactive Virtual Network Embedding in Kubernetes Using Transformer-Based Reinforcement Learning for IIoT

KubePredict-VNE is a proactive Kubernetes-aware Virtual Network Embedding (VNE) framework integrating Transformer-based workload forecasting, Graph Attention Networks (GAT), and Proximal Policy Optimization (PPO) for intelligent orchestration in dynamic IIoT and cloud-native environments.

The framework models proactive node-link embedding under dynamic workloads, Kubernetes-aware constraints, resource reservation, and topology-aware orchestration.

---

# Key Features

• Multi-horizon Transformer-based workload forecasting
• Graph Attention Network (GAT) substrate encoding
• PPO-based proactive orchestration
• Kubernetes-aware constraint handling
• Joint node-link embedding
• Forecast-aware proactive reservation
• Fragmentation-aware reward optimization
• Dynamic VNR lifecycle management
• Node failure resilience modeling
• Resource recovery and re-embedding support

---

# Framework Architecture

The proposed framework integrates:

1. Transformer Forecaster
   Predicts future VNR workload demand and reservation requirements.

2. GAT Encoder
   Learns topology-aware substrate graph representations.

3. PPO Agent
   Performs sequential constrained embedding decisions.

4. Kubernetes-Aware Environment
   Models affinity, anti-affinity, taints/tolerations, direct-link constraints, dynamic arrivals, failures, and resource reservation.

---

# Repository Structure

```text
KubePredict-VNE/
│
├── environment/
│   └── env.py
│
├── models/
│   ├── gat_encoder.py
│   ├── ppo_agent.py
│   └── transformer_forecaster.py
│
├── training/
│   └── trainer.py
│
├
│
├── results/
│   ├── csv/
│   └── figures/
│
├── config.py
├── main.py
└── README.md
```

---

# Installation

Clone the repository:

```bash
https://github.com/Ali121472536/KubePredict-VNE-Proactive-Virtual-Network-Embedding-in-Kubernetes-.git
```

Create environment:

```bash
conda create -n kubepredict python=3.10
conda activate kubepredict
```

Install dependencies:

```bash
pip install torch
pip install torch-geometric
pip install networkx
pip install numpy
pip install matplotlib
pip install pandas
```

---

# Running the Framework

Train the framework:

```bash
python main.py
```

The framework automatically performs:

• Dynamic VNR generation
• PPO training
• Proactive reservation
• Forecast-aware orchestration
• Resource recovery
• Evaluation metric generation

---

# Evaluation Metrics

The framework evaluates:

• Slice Acceptance Ratio (SAR)
• Revenue-to-Cost Ratio (RCR)
• CPU Utilization
• Bandwidth Utilization
• Training Convergence
• Inference Latency
• Multi-horizon Forecasting Accuracy
• Node Failure Resilience

---

# Experimental Configuration

Default configuration:

• Substrate nodes: 80
• VNR size: 3–6 virtual nodes
• PPO episodes: 3000
• Forecast horizon: 12
• GAT hidden dimension: 64
• PPO learning rate: 1e-4

All parameters can be modified in:

```text
config.py
```

---

# Reproducibility

This repository is released for research reproducibility purposes.

Due to stochastic PPO training, random topology generation, and dynamic workload sampling, numerical results may vary slightly across executions.

The implementation is a research-oriented simulator and does not represent a production Kubernetes deployment.

---

# Citation

If you use this repository in your research, please cite:

KubePredict-VNE: Proactive Virtual Network
Embedding in Kubernetes Using Transformer-Based
Reinforcement Learning for IIoT

# License

This project is released for academic and research purposes only.

---

# Contact

Email: q.ali1038@gmail.com

