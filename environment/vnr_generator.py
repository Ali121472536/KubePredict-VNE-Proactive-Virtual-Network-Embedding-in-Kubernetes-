# environment/vnr_generator.py

import networkx as nx
import random

from config import *


def generate_vnr():

    num_nodes = random.randint(
        VNR_NODES_MIN,
        VNR_NODES_MAX
    )

    G = nx.erdos_renyi_graph(
        num_nodes,
        0.5
    )

    # ======================================================
    # NODE ATTRIBUTES
    # ======================================================

    for node in G.nodes:

        G.nodes[node]['cpu'] = random.randint(
            VNR_CPU_MIN,
            VNR_CPU_MAX
        )

        # ==============================================
        # KUBERNETES AFFINITY GROUP
        # ==============================================

        G.nodes[node]['affinity_group'] = \
            random.randint(0, 2)

        # ==============================================
        # ANTI-AFFINITY FLAG
        # ==============================================

        G.nodes[node]['anti_affinity'] = \
            random.choice([True, False])

        # ==============================================
        # TOLERATION SUPPORT
        # ==============================================

        G.nodes[node]['tolerates_taint'] = \
            random.choice([True, False])

    # ======================================================
    # LINK ATTRIBUTES
    # ======================================================

    for u, v in G.edges:

        G.edges[u, v]['bw'] = random.randint(
            VNR_BW_MIN,
            VNR_BW_MAX
        )

    return G