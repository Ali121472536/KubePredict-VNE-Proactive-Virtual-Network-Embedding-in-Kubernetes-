# environment/env.py

import numpy as np
import networkx as nx

from environment.substrate import generate_substrate
from environment.vnr_generator import generate_vnr


class VNEEnvironment:

    # ======================================================
    # INITIALIZATION
    # ======================================================

    def __init__(self):

        self.substrate = generate_substrate()

        self.current_vnr = None

        # ==============================================
        # ACTIVE EMBEDDED VNRS
        # ==============================================

        self.active_embeddings = []

        # ==============================================
        # GLOBAL PENALTIES
        # ==============================================

        self.direct_link_penalty = 0

        self.affinity_penalty = 0

        # ==============================================
        # SEQUENTIAL ORCHESTRATION
        # ==============================================

        self.max_requests = 100

        self.current_request = 0

        self.accepted_requests = 0

    # ======================================================
    # RESET ENVIRONMENT
    # ======================================================

    def reset(self):

        # ==============================================
        # NEW EPISODE
        # ==============================================

        self.substrate = generate_substrate()

        self.current_request = 0

        self.accepted_requests = 0

        # ==============================================
        # CLEAR ACTIVE EMBEDDINGS
        # ==============================================

        self.active_embeddings = []

        # ==============================================
        # GENERATE FIRST REQUEST
        # ==============================================

        self.current_vnr = generate_vnr()

        return self.get_state()

    # ======================================================
    # RELEASE EXPIRED VNRS
    # ======================================================

    def release_expired_vnrs(self):

        remaining_embeddings = []

        for embedding in self.active_embeddings:

            # ==========================================
            # DECREASE LIFETIME
            # ==========================================

            embedding['lifetime'] -= 1

            # ==========================================
            # STILL ACTIVE
            # ==========================================

            if embedding['lifetime'] > 0:
                remaining_embeddings.append(
                    embedding
                )

                continue

            # ==========================================
            # RELEASE NODE CPU
            # ==========================================

            for vnode, snode in \
                    embedding['node_mapping'].items():
                cpu_req = embedding['vnr'].nodes[
                    vnode
                ]['cpu']

                self.substrate.nodes[snode][
                    'cpu_available'
                ] += cpu_req

                # ======================================
                # CLAMP TO MAX CAPACITY
                # ======================================

                self.substrate.nodes[snode][
                    'cpu_available'
                ] = min(

                    self.substrate.nodes[snode][
                        'cpu_available'
                    ],

                    self.substrate.nodes[snode][
                        'cpu'
                    ]
                )

            # ==========================================
            # RELEASE LINK BANDWIDTH
            # ==========================================

            for (u, v), path in \
                    embedding['link_mapping'].items():

                bw_req = embedding['vnr'].edges[
                    u,
                    v
                ]['bw']

                for i in range(len(path) - 1):
                    a = path[i]

                    b = path[i + 1]

                    self.substrate.edges[a, b][
                        'bw_available'
                    ] += bw_req

                    # ==================================
                    # CLAMP TO MAX CAPACITY
                    # ==================================

                    self.substrate.edges[a, b][
                        'bw_available'
                    ] = min(

                        self.substrate.edges[a, b][
                            'bw_available'
                        ],

                        self.substrate.edges[a, b][
                            'bw'
                        ]
                    )

        # ==============================================
        # UPDATE ACTIVE EMBEDDINGS
        # ==============================================

        self.active_embeddings = \
            remaining_embeddings

    # ======================================================
    # STATE REPRESENTATION
    # ======================================================

    def get_state(self):
        cpu_state = [

            self.substrate.nodes[n]['cpu_available']

            for n in self.substrate.nodes
        ]

        return np.array(
            cpu_state,
            dtype=np.float32
        )

    # ======================================================
    # WORKLOAD HISTORY FOR TRANSFORMER
    # ======================================================

    def get_workload_sequence(self):

        sequence = []

        for _ in range(5):

            total_cpu = 0

            for vnode in self.current_vnr.nodes:

                total_cpu += self.current_vnr.nodes[
                    vnode
                ]['cpu']

            sequence.append(total_cpu)

        return sequence

    # ======================================================
    # PROACTIVE RESOURCE RESERVATION
    # ======================================================

    def reserve_future_resources(
            self,
            predicted_demand
    ):

        reserve_ratio = min(
            predicted_demand / 100.0,
            0.15
        )

        # ==============================================
        # RESERVE NODE CPU
        # ==============================================

        for node in self.substrate.nodes:

            cpu_avail = self.substrate.nodes[node][
                'cpu_available'
            ]

            reserved_cpu = int(
                cpu_avail * reserve_ratio
            )

            self.substrate.nodes[node][
                'cpu_reserved'
            ] = reserved_cpu

        # ==============================================
        # RESERVE LINK BANDWIDTH
        # ==============================================

        for u, v in self.substrate.edges:

            bw_avail = self.substrate.edges[u, v][
                'bw_available'
            ]

            reserved_bw = int(
                bw_avail * reserve_ratio
            )

            self.substrate.edges[u, v][
                'bw_reserved'
            ] = reserved_bw

    # ======================================================
    # KUBERNETES FEASIBILITY CHECK
    # ======================================================

    def check_kubernetes_constraints(
            self,
            vnode,
            snode,
            node_mapping
    ):

        vnode_data = self.current_vnr.nodes[vnode]

        snode_data = self.substrate.nodes[snode]

        # ==============================================
        # TAINT PENALTY
        # ==============================================

        if (
            snode_data['tainted']
            and
            not vnode_data['tolerates_taint']
        ):

            self.affinity_penalty += 5

        # ==============================================
        # SOFT ANTI-AFFINITY
        # ==============================================

        if vnode_data['anti_affinity']:

            for mapped_vnode, mapped_snode in \
                    node_mapping.items():

                mapped_data = \
                    self.current_vnr.nodes[mapped_vnode]

                if (
                    mapped_data['affinity_group']
                    ==
                    vnode_data['affinity_group']
                    and
                    mapped_snode == snode
                ):

                    self.affinity_penalty += 4

        # ==============================================
        # SOFT AFFINITY PREFERENCE
        # ==============================================

        for mapped_vnode, mapped_snode in \
                node_mapping.items():

            mapped_data = \
                self.current_vnr.nodes[mapped_vnode]

            if (
                mapped_data['affinity_group']
                ==
                vnode_data['affinity_group']
            ):

                if (
                    self.substrate.nodes[
                        mapped_snode
                    ]['zone']

                    !=

                    snode_data['zone']
                ):

                    self.affinity_penalty += 3

        return True

    # ======================================================
    # PPO-BASED NODE MAPPING
    # ======================================================

    def map_virtual_nodes(self, first_action):

        node_mapping = {}

        virtual_nodes = list(
            self.current_vnr.nodes
        )

        # ==============================================
        # FIRST NODE
        # ==============================================

        first_vnode = virtual_nodes[0]

        cpu_req = self.current_vnr.nodes[
            first_vnode
        ]['cpu']

        cpu_avail = (

            self.substrate.nodes[first_action][
                'cpu_available'
            ]

            -

            self.substrate.nodes[first_action][
                'cpu_reserved'
            ]
        )

        if (
            cpu_avail < cpu_req
            or
            not self.check_kubernetes_constraints(
                first_vnode,
                first_action,
                node_mapping
            )
        ):

            found = False

            for snode in self.substrate.nodes:

                cpu_avail = (

                    self.substrate.nodes[snode][
                        'cpu_available'
                    ]

                    -

                    self.substrate.nodes[snode][
                        'cpu_reserved'
                    ]
                )

                if (
                    cpu_avail >= cpu_req
                    and self.check_kubernetes_constraints(
                        first_vnode,
                        snode,
                        node_mapping
                    )
                ):

                    first_action = snode

                    found = True

                    break

            if not found:

                return None

        node_mapping[first_vnode] = first_action

        # ==============================================
        # REMAINING NODES
        # ==============================================

        for vnode in virtual_nodes[1:]:

            cpu_req = self.current_vnr.nodes[
                vnode
            ]['cpu']

            mapped = False

            for snode in self.substrate.nodes:

                cpu_avail = (

                    self.substrate.nodes[snode][
                        'cpu_available'
                    ]

                    -

                    self.substrate.nodes[snode][
                        'cpu_reserved'
                    ]
                )

                if (
                    cpu_avail >= cpu_req
                    and self.check_kubernetes_constraints(
                        vnode,
                        snode,
                        node_mapping
                    )
                ):

                    node_mapping[vnode] = snode

                    mapped = True

                    break

            if not mapped:

                return None

        return node_mapping

    # ======================================================
    # LINK MAPPING
    # ======================================================

    def map_virtual_links(self, node_mapping):

        link_mapping = {}

        for (u, v) in self.current_vnr.edges:

            bw_req = self.current_vnr.edges[u, v]['bw']

            src = node_mapping[u]

            dst = node_mapping[v]

            try:

                path = nx.shortest_path(
                    self.substrate,
                    source=src,
                    target=dst
                )

            except nx.NetworkXNoPath:

                return None

            # ==============================================
            # DIRECT LINK PENALTY
            # ==============================================

            if len(path) > 3:

                self.direct_link_penalty += (
                    len(path) - 3
                ) * 2

            # ==============================================
            # BANDWIDTH FEASIBILITY
            # ==============================================

            feasible = True

            for i in range(len(path) - 1):

                a = path[i]

                b = path[i + 1]

                bw_avail = (

                    self.substrate.edges[a, b][
                        'bw_available'
                    ]

                    -

                    self.substrate.edges[a, b][
                        'bw_reserved'
                    ]
                )

                if bw_avail < bw_req * 0.7:

                    feasible = False

                    break

            if not feasible:

                return None

            link_mapping[(u, v)] = path

        return link_mapping

    # ======================================================
    # RESOURCE ALLOCATION
    # ======================================================

    def allocate_resources(
            self,
            node_mapping,
            link_mapping
    ):

        # ==============================================
        # NODE CPU
        # ==============================================

        for vnode, snode in node_mapping.items():

            cpu_req = self.current_vnr.nodes[
                vnode
            ]['cpu']

            self.substrate.nodes[snode][
                'cpu_available'
            ] = max(
                0,
                self.substrate.nodes[snode][
                    'cpu_available'
                ] - cpu_req
            )

        # ==============================================
        # LINK BANDWIDTH
        # ==============================================

        for (u, v), path in link_mapping.items():

            bw_req = self.current_vnr.edges[
                u,
                v
            ]['bw']

            for i in range(len(path) - 1):

                a = path[i]

                b = path[i + 1]

                self.substrate.edges[a, b][
                    'bw_available'
                ] = max(
                    0,
                    self.substrate.edges[a, b][
                        'bw_available'
                    ] - bw_req
                )

    # ======================================================
    # COMPUTE FRAGMENTATION
    # ======================================================

    def compute_fragmentation(self):

        available_resources = []

        for node in self.substrate.nodes:

            cpu = self.substrate.nodes[node][
                'cpu_available'
            ]

            cpu = max(cpu, 0)

            available_resources.append(cpu)

        total = sum(available_resources)

        if total <= 0:

            return 1.0

        probs = np.array(
            available_resources,
            dtype=np.float32
        ) / (total + 1e-9)

        probs = np.clip(
            probs,
            1e-9,
            1.0
        )

        entropy = -np.sum(
            probs * np.log(probs)
        )

        normalized_entropy = entropy / np.log(
            len(available_resources)
        )

        if np.isnan(normalized_entropy):

            return 1.0

        return normalized_entropy

    # ======================================================
    # OVER-RESERVATION PENALTY
    # ======================================================

    def compute_overreservation_penalty(self):

        total_reserved = 0

        for node in self.substrate.nodes:

            total_reserved += self.substrate.nodes[
                node
            ]['cpu_reserved']

        penalty = total_reserved * 0.05

        return penalty

    # ======================================================
    # STEP FUNCTION
    # ======================================================

    def step(
            self,
            action,
            predicted_demand
    ):

        # ==============================================
        # RESET PENALTIES
        # ==============================================

        self.direct_link_penalty = 0

        self.affinity_penalty = 0

        # ==============================================
        # RESERVATION
        # ==============================================

        self.reserve_future_resources(
            predicted_demand
        )

        # ==============================================
        # NODE MAPPING
        # ==============================================

        node_mapping = self.map_virtual_nodes(
            action
        )

        if node_mapping is None:

            reward = -20

            self.current_request += 1

            self.current_vnr = generate_vnr()

            done = False

            if self.current_request >= self.max_requests:

                done = True

            return (
                self.get_state(),
                reward,
                done
            )

        # ==============================================
        # LINK MAPPING
        # ==============================================

        link_mapping = self.map_virtual_links(
            node_mapping
        )

        if link_mapping is None:

            reward = -20

            self.current_request += 1

            self.current_vnr = generate_vnr()

            done = False

            if self.current_request >= self.max_requests:

                done = True

            return (
                self.get_state(),
                reward,
                done
            )

        # ==============================================
        # LATENCY PENALTY
        # ==============================================

        total_hops = 0

        for path in link_mapping.values():

            total_hops += len(path) - 1

        latency_penalty = total_hops * 2

        # ==============================================
        # REVENUE
        # ==============================================

        revenue = 0

        for vnode in self.current_vnr.nodes:

            revenue += self.current_vnr.nodes[
                vnode
            ]['cpu']

        for u, v in self.current_vnr.edges:

            revenue += self.current_vnr.edges[
                u,
                v
            ]['bw']

        # ==============================================
        # COST
        # ==============================================

        cost = revenue * 0.4

        # ==============================================
        # ALLOCATE RESOURCES
        # ==============================================

        self.allocate_resources(
            node_mapping,
            link_mapping
        )

        # ==============================================
        # FRAGMENTATION
        # ==============================================

        fragmentation = \
            self.compute_fragmentation()

        # ==============================================
        # RESERVATION PENALTY
        # ==============================================

        reservation_penalty = \
            self.compute_overreservation_penalty()

        # ==============================================
        # FINAL REWARD
        # ==============================================

        reward = (

            revenue

            - 0.3 * cost

            - 0.2 * fragmentation

            - reservation_penalty

            - latency_penalty

            - self.direct_link_penalty

            - self.affinity_penalty
        )

        reward = np.clip(
            reward,
            -50,
            3000
        )

        # ==============================================
        # ACCEPTED REQUEST
        # ==============================================

        self.accepted_requests += 1

        # ==============================================
        # NEXT REQUEST
        # ==============================================

        self.current_request += 1

        self.current_vnr = generate_vnr()

        # ==============================================
        # TERMINATION
        # ==============================================

        done = False

        if self.current_request >= self.max_requests:

            done = True

        return (
            self.get_state(),
            reward,
            done
        )

    # ======================================================
    # ACCEPTANCE RATIO
    # ======================================================

    def get_acceptance_ratio(self):

        if self.current_request == 0:

            return 0

        return (
            self.accepted_requests
            /
            self.current_request
        )