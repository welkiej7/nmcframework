"""
Title: Latent and Observed Network Generation from a User Similarity Matrix
Date Created: 02.12.2024
Date Last Updated: 05.12.2025


Notes
-----

If you are using please cite the corresponding paper on GitHub. 

"""

import numpy as np
from termcolor import colored
from numpy.typing import NDArray
from opinionspaces import calculate_similarity_on_single_topic

def generate_latent_network(
    similarity_matrix: NDArray,
    m: int = 2,
    alpha: float = 1.0,
    beta: float = 1.0,
    verbose: bool = True,
):
    """
    Generate an edge list using a Karimi-style weighted attachment mechanism
    based on degree and homophily (similarity).

    Parameters
    ----------
    similarity_matrix : ndarray of shape (n_nodes, n_nodes)
        Pairwise similarity between nodes. Entry (i, j) gives the similarity
        between node i and node j.
    m : int, default=2
        Number of edges each new node will create (if possible).
    alpha : float, default=1.0
        Exponent controlling the strength of preferential attachment by degree.
    beta : float, default=1.0
        Exponent controlling the strength of homophily (similarity-based attachment).
    verbose : bool, default=True
        If True, print detailed information about probabilities and choices.

    Returns
    -------
    edge_list : list of tuple[int, int]
        List of edges (u, v) in the generated network.
    """
    n = similarity_matrix.shape[0]
    nodes = np.arange(n)

 
    degrees = np.zeros(n, dtype=int)
    edge_list: list[tuple[int, int]] = []


    st1, st2 = np.random.choice(nodes, size=2, replace=False)
    edge_list.append((int(st1), int(st2)))
    degrees[st1] = 1
    degrees[st2] = 1

    existing_nodes = [int(st1), int(st2)]

    remaining_nodes = np.setdiff1d(nodes, [st1, st2], assume_unique=True)
    nodes_order = np.random.permutation(remaining_nodes)

    for node in nodes_order:
        node = int(node)

        if verbose:
            print(f"{'-' * 20}\nNode {colored(node, 'red')} is casting.\n{'*' * 20}")

        
        existing_arr = np.array(existing_nodes, dtype=int)

        
        existing_degrees = degrees[existing_arr].astype(float)
        degree_probabilities = existing_degrees / existing_degrees.sum()

        if verbose:
            print(
                f"Highest degree probability is "
                f"{colored(np.max(degree_probabilities), 'yellow')}."
                f" Lowest probability is "
                f"{colored(np.min(degree_probabilities), 'green')}\n"
                f"{'*' * 20}"
            )

        
        existing_nodes_homophily = similarity_matrix[existing_arr, node].astype(float)
        homophily_probabilities = (
            existing_nodes_homophily / existing_nodes_homophily.sum()
        )

        if verbose:
            print(
                f"Highest homophily probability is "
                f"{colored(np.max(homophily_probabilities), 'yellow')}."
                f" Lowest probability is "
                f"{colored(np.min(homophily_probabilities), 'green')}\n"
                f"{'*' * 20}"
            )

        
        degree_probabilities **= alpha
        homophily_probabilities **= beta


        probabilities = degree_probabilities * homophily_probabilities
        probabilities /= probabilities.sum()

        if verbose:
            print(
                f"Highest probability is "
                f"{colored(np.max(probabilities), 'yellow')}."
                f" Lowest probability is "
                f"{colored(np.min(probabilities), 'green')}\n"
                f"{'*' * 20}"
            )


        k = min(m, len(existing_nodes))

        picked_indices = np.random.choice(
            existing_arr, p=probabilities, replace=False, size=k
        )

        if verbose:
            print(
                f"Picked {colored(picked_indices, 'red')}."
                f" Their probabilities were {colored(probabilities, 'blue')}"
            )


        degrees[picked_indices] += 1
        degrees[node] += k


        edge_list.extend((int(node), int(p)) for p in picked_indices)


        existing_nodes.append(node)

        if verbose:
            print("Existing Nodes:", existing_nodes)

    return edge_list



def calculate_similarity_single_topic(opinion_array: np.ndarray, topic_idx: int) -> np.ndarray:
    """
    Compute a pairwise similarity matrix based on a single opinion topic.

    The similarity between two nodes i and j is defined as:
        sim(i, j) = 1 - |x_i - x_j| / 2

    where x_i and x_j are the opinion values of nodes i and j on the selected
    topic. This maps differences in the range [-2, 2] to similarities in [0, 1].

    Parameters
    ----------
    opinion_array : np.ndarray, shape (N, T)
        Opinion matrix where N is the number of nodes and T is the number of topics.
    topic_idx : int
        Index of the topic used to compute similarity.

    Returns
    -------
    np.ndarray, shape (N, N)
        Pairwise similarity matrix for the selected topic with values in [0, 1].
    """
    topic_vals = opinion_array[:, topic_idx]
    diff = np.abs(topic_vals[:, None] - topic_vals[None, :])
    sim = 1.0 - diff / 2.0
    return sim


def generate_interaction_network_from_edgelist(
    latent_edgelist,
    opinion_array: np.ndarray,
    topic_idx: int,
    rt_min: int = 1,
    rt_max: int = 3,
    latent: float = 1.0,
    beta: float = 2.0,
    verbose: bool = True) -> list:

    
    """
    Generate a directed interaction network from a latent undirected edge list
    and a single-topic opinion similarity mechanism.

    Each node initiates a random number of interactions drawn uniformly from
    [rt_min, rt_max]. For each interaction, with probability `latent`, the target
    node is selected from the latent network neighbors. With probability
    (1 - latent), the target is selected according to a similarity-weighted
    probability distribution based on a single opinion topic.

    Parameters
    ----------
    latent_edgelist : list of tuple (u, v)
        Undirected latent network represented as a list of node index pairs.
    opinion_array : np.ndarray, shape (N, T)
        Opinion matrix where N is the number of nodes and T is the number of topics.
    topic_idx : int
        Index of the topic used for similarity-based interactions.
    rt_min : int, default=1
        Minimum number of interactions each node initiates.
    rt_max : int, default=3
        Maximum number of interactions each node initiates.
    latent : float, default=1.0
        Probability of selecting an interaction partner from the latent network.
    beta : float, default=2.0
        Exponent applied to similarity values to control selection sharpness.
    verbose : bool, default=True
        If True, prints diagnostic information during interaction generation.

    Returns
    -------
    list of tuple (source, target)
        Directed interaction edge list where each tuple represents an interaction
        from `target` to `source`.
    """
    num_nodes = opinion_array.shape[0]

    if latent < 0 or latent > 1:
        raise ValueError("latent must be a probability in [0, 1].")

    if rt_min < 0 or rt_max < rt_min:
        raise ValueError("Require 0 <= rt_min <= rt_max.")

    if topic_idx < 0 or topic_idx >= opinion_array.shape[1]:
        raise ValueError("topic_idx out of bounds for opinion_array.")

    neighbors = {i: [] for i in range(num_nodes)}
    for u, v in latent_edgelist:
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            neighbors[u].append(v)
            neighbors[v].append(u)
        else:
            raise ValueError(f"Edge ({u}, {v}) has node id outside [0, {num_nodes-1}].")

    rts_total = np.random.randint(rt_min, rt_max + 1, size=num_nodes)
    similarity_on_opinion = calculate_similarity_single_topic(opinion_array, topic_idx).astype(float)
    np.fill_diagonal(similarity_on_opinion, 0.0)

    interaction_edges = []

    for node_index, rt in enumerate(rts_total):
        for _ in range(rt):

            use_latent = np.random.rand() < latent

            if use_latent:
                node_neighbors = neighbors[node_index]

                if len(node_neighbors) != 0:
                    picked_node = np.random.choice(node_neighbors)

                    if verbose:
                        print(f"{node_index} picked {picked_node} via LATENT.")

                    interaction_edges.append((picked_node, node_index))
                    continue

            probs = similarity_on_opinion[:, node_index].copy()
            probs[node_index] = 0.0

            if probs.sum() <= 0:
                probs = np.ones(num_nodes)
                probs[node_index] = 0.0

            probs = probs ** beta
            probs_sum = probs.sum()

            if probs_sum <= 0:
                probs = np.ones(num_nodes)
                probs[node_index] = 0.0
                probs_sum = probs.sum()

            probs = probs / probs_sum
            picked_node = np.random.choice(np.arange(num_nodes), p=probs)

            if verbose:
                max_p = np.max(probs)
                nonzero = probs[probs > 0]
                min_p = np.min(nonzero) if nonzero.size > 0 else 0.0
                print(
                    f"{node_index} picked {picked_node} via TOPIC {topic_idx}.\n"
                    f"Diff to highest prob: {abs(max_p - probs[picked_node])}\n"
                    f"Diff to lowest non-zero prob: {abs(min_p - probs[picked_node])}"
                )

            interaction_edges.append((picked_node, node_index))

    return interaction_edges


