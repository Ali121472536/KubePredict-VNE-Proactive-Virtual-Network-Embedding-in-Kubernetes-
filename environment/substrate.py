# environment/substrate.py

import networkx as nx
import random

from config import *


def generate_substrate():
    while True:

        G = nx.erdos_renyi_graph(
            NUM_SUBSTRATE_NODES,
            0.3
        )

        if nx.is_connected(G):
            break

    # ======================================================
    # INITIALIZE SUBSTRATE NODES
    # ======================================================

    for node in G.nodes:

        cpu_capacity = random.randint(
            SUBSTRATE_CPU_MIN,
            SUBSTRATE_CPU_MAX
        )

        # ==============================================
        # CPU RESOURCES
        # ==============================================

        G.nodes[node]['cpu'] = cpu_capacity

        G.nodes[node]['cpu_available'] = (
            cpu_capacity
        )

        # ==============================================
        # PROACTIVE RESERVED CPU
        # ==============================================

        G.nodes[node]['cpu_reserved'] = 0

        # ==============================================
        # KUBERNETES TAINT LABEL
        # ==============================================

        G.nodes[node]['tainted'] = random.choice(
            [True, False]
        )

        # ==============================================
        # KUBERNETES ZONE / CLUSTER REGION
        # ==============================================

        G.nodes[node]['zone'] = random.randint(
            0,
            2
        )

    # ======================================================
    # INITIALIZE SUBSTRATE LINKS
    # ======================================================

    for u, v in G.edges:

        bw_capacity = random.randint(
            SUBSTRATE_BW_MIN,
            SUBSTRATE_BW_MAX
        )

        # ==============================================
        # BANDWIDTH RESOURCES
        # ==============================================

        G.edges[u, v]['bw'] = bw_capacity

        G.edges[u, v]['bw_available'] = (
            bw_capacity
        )

        # ==============================================
        # PROACTIVE RESERVED BANDWIDTH
        # ==============================================

        G.edges[u, v]['bw_reserved'] = 0

        # ==============================================
        # DIRECT-LINK PRIORITY
        # ==============================================

        G.edges[u, v]['direct_link'] = random.choice(
            [True, False]
        )

    return G