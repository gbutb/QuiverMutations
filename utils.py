"""
utils.py
--------
"""
from tqdm import trange

from copy import deepcopy

import numpy as np
# from agent import Agent
from environment import Environment


def train_agent(env: Environment, agent, num_epochs = 1024, max_steps = 32, batch_size = 128, min_memory_size = 1024, verbosity = 0) -> np.ndarray:
    """
    @return Array of (epochs, loss_vals). Shape: (Num_epochs, 2).
    """
    steps_to_update_target_model = 0
    epochs = trange(num_epochs) if verbosity > 0 else range(num_epochs)
    loss_vals = []

    for epoch in epochs:
        state = env.random_state()
        terminated = False
        total_loss = 0.0
        count = 0

        for _ in range(max_steps):
            steps_to_update_target_model += 1
            action = agent.act(state)
            new_state, (fitness, done) = env.act(state, action)

            reward = 0
            if done:
                reward = 10
                terminated = True
            else:
                reward = -1. + (fitness)# - env.fitness(state)[0])

            agent.store(state, action, reward, new_state, terminated)

            if terminated or (steps_to_update_target_model % 4 == 0):
                if agent.current_memory_size > min_memory_size:
                    history = agent.replay(batch_size)
                    total_loss += history.history['loss'][-1]
                    count += 1

            # Move
            state = deepcopy(new_state)

            if terminated:
                if steps_to_update_target_model >= 100:
                    agent.update_target_model()
                    steps_to_update_target_model = 0

            if terminated:
                break
        if verbosity > 0:
            epochs.set_description(f"Epoch {epoch}, loss = {total_loss / (count+1e-10):e}")
        if agent.current_memory_size > min_memory_size:
            loss_vals.append([epoch, total_loss / (count+1e-10)])
    return np.asarray(loss_vals)


def walk_agent(state, env: Environment, agent, max_steps = 32):
    path = [state]
    actions = []
    for _ in range(max_steps):
        action = agent.act(state)
        state, (_, done) = env.act(state, action)
        path.append(state)
        actions.append(action)
        if done:
            break
    return path, actions


import networkx as nx
import itertools as it
def draw_labeled_multigraph(G, attr_name, ax=None):
    """
    Source: https://networkx.org/documentation/stable/auto_examples/drawing/plot_multigraphs.html
    """
    # Works with arc3 and angle3 connectionstyles
    connectionstyle = [f"arc3,rad={r}" for r in it.accumulate([0.15] * 4)]
    # connectionstyle = [f"angle3,angleA={r}" for r in it.accumulate([30] * 4)]

    pos = nx.shell_layout(G)
    nx.draw_networkx_nodes(G, pos, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=20, ax=ax)
    nx.draw_networkx_edges(
        G, pos, edge_color="black", alpha=0.5, connectionstyle=connectionstyle, ax=ax
    )

    labels = {
        tuple(edge): f"{attrs[attr_name]}"
        for *edge, attrs in G.edges(keys=True, data=True)
    }
    nx.draw_networkx_edge_labels(
        G,
        pos,
        labels,
        connectionstyle=connectionstyle,
        label_pos=0.3,
        font_color="blue",
        bbox={"alpha": 0},
        ax=ax)
