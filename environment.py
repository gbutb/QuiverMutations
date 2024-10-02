import abc
from typing import Tuple, List

import numpy as np
import networkx as nx

class Environment(object):
    @abc.abstractmethod
    def random_state(self):
        """
        Generates a random state.
        """

    @abc.abstractmethod
    def fitness(self, state) -> Tuple[float, bool]:
        """Computes fitness function.

        @param state: Specifies current state.

        @return Tuple (reward, terminated).
        """

    @abc.abstractmethod
    def act(self, state, action: int) -> Tuple[List, float]:
        """
        @param state: Current state.
        @param action: Index of the action using which to act on the state.

        @return Tuple (next state, fitness of the next state).
        """

    @property
    @abc.abstractmethod
    def num_actions(self):
        """
        @return Number of actions.
        """

class QuiverMutationEnvironment(Environment):
    def __init__(self, quiver, max_edges = None, random_max_steps = 4):
        """
        @param quiver: adjacency matrix of quiver
        """
        self._quiver = quiver
        self._num_nodes = quiver.shape[0]
        self._max_edges = max_edges
        self._random_max_steps = random_max_steps

    def random_state(self):
        actions = np.random.choice(self.num_actions, size=(self._random_max_steps,))
        mutated = self._quiver.reshape(-1).copy()
        for a in actions:
            mutated, _ = self.act(mutated, a)
        return mutated.reshape(-1)

    @staticmethod
    def _mutate_mat(adjacency_mat, node):
        # adjacency_mat = adjacency_mat.copy()
        # for i in range(adjacency_mat.shape[0]):
        #     if i == node: continue
        #     if adjacency_mat[i][node] != 0:
        #         for j in range(adjacency_mat.shape[0]):
        #             if j == node or i == j: continue
        #             if adjacency_mat[node][j] != 0:
        #                 if adjacency_mat[j][i] == 0:
        #                     # print(f"Add i={i+1} j={j+1} node={node+1}")
        #                     adjacency_mat[i][j] += 1
        #                 else:
        #                     # print(f"Remove i={i+1} j={j+1} node={node+1}")
        #                     adjacency_mat[j][i] = max(adjacency_mat[j][i]-1, 0)
        #                 # print("ITER")

        # temp = adjacency_mat[node,...].copy()
        # adjacency_mat[node,...] = adjacency_mat[...,node]
        # adjacency_mat[...,node] = temp
        # return adjacency_mat
        mask = np.zeros_like(adjacency_mat)
        mask[node, ...] += 1
        mask[..., node] += 1
        mask = mask > 0

        mat= mask*(-adjacency_mat) + (~mask)*(adjacency_mat + \
            (np.abs(adjacency_mat[..., node][..., np.newaxis])*adjacency_mat[node, ...][np.newaxis, ...] + \
            adjacency_mat[..., node][..., np.newaxis]*np.abs(adjacency_mat[node, ...][np.newaxis, ...]))//2)
        return mat
    

    def fitness(self, state):
        state_ = state.reshape(self._num_nodes, self._num_nodes).copy()
        # There's a faster way...
        # rows, cols = np.where(state > 0)
        # edges = zip(rows.tolist(), cols.tolist())
        state_[state_ <= 0] = 0
        gr = nx.from_numpy_array(state_, create_using=nx.MultiDiGraph())
        # gr = nx.DiGraph()
        # gr.add_edges_from(edges)
        return -len([*nx.simple_cycles(gr)])

    def act(self, state, action: int) -> Tuple[List, float]:
        state = state.reshape(self._num_nodes, self._num_nodes)
        next_state = self._mutate_mat(state, action).reshape(-1)

        if self._max_edges is not None:
            if np.abs(next_state).sum() > 2*self._max_edges:
                next_state = state.reshape(-1)
        fitness = self.fitness(next_state)
        return next_state, (fitness, fitness == 0)

    @property
    def num_actions(self):
        return self._num_nodes
