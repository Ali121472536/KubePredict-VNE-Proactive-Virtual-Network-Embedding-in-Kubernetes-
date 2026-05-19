# config.py

# ======================================================
# SUBSTRATE NETWORK CONFIGURATION
# ======================================================

NUM_SUBSTRATE_NODES = 80

SUBSTRATE_CPU_MIN = 90
SUBSTRATE_CPU_MAX = 130

# Reduced bandwidth capacity
# improves BW utilization realism

SUBSTRATE_BW_MIN = 120
SUBSTRATE_BW_MAX = 250

# ======================================================
# VIRTUAL NETWORK REQUEST (VNR)
# ======================================================

VNR_NODES_MIN = 3
VNR_NODES_MAX = 6

VNR_CPU_MIN = 15
VNR_CPU_MAX = 30

# Moderate bandwidth demand

VNR_BW_MIN = 30
VNR_BW_MAX = 80

# ======================================================
# TRAINING CONFIGURATION
# ======================================================

MAX_EPISODES = 3000

LEARNING_RATE = 1e-4

GAMMA = 0.99

PPO_EPS_CLIP = 0.2

# ======================================================
# TRANSFORMER FORECASTING
# ======================================================

FORECAST_HORIZON = 12

WORKLOAD_SEQUENCE_LENGTH = 5

TRANSFORMER_DMODEL = 64

TRANSFORMER_HEADS = 4

TRANSFORMER_LAYERS = 2

# ======================================================
# REWARD PARAMETERS
# ======================================================

LAMBDA_COST = 0.15

LAMBDA_FRAGMENTATION = 0.08

LAMBDA_RESERVATION = 0.05

LAMBDA_FORECAST = 0.15

# ======================================================
# KUBERNETES CONSTRAINTS
# ======================================================

ENABLE_TAINTS = True

ENABLE_ANTI_AFFINITY = True

ENABLE_AFFINITY = True

ENABLE_DIRECT_LINKS = True

# ======================================================
# RESERVATION SETTINGS
# ======================================================

MAX_RESERVATION_RATIO = 0.20

OVER_RESERVATION_PENALTY = 0.01