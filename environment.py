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
    def __init__(self, quiver):
        """
        @param quiver: adjacency matrix of quiver
        """
        self._quiver = quiver
        self._num_nodes = quiver.shape[0]

    def random_state(self):
        return self._quiver.reshape(-1)

    @staticmethod
    def _mutate_mat(adjacency_mat, node):
        adjacency_mat = adjacency_mat.copy()
        for i in range(adjacency_mat.shape[0]):
            if i == node: continue
            if adjacency_mat[i][node] != 0:
                for j in range(adjacency_mat.shape[0]):
                    if j == node: continue
                    if adjacency_mat[node][j] != 0:
                        if adjacency_mat[j][i] == 0:
                            adjacency_mat[i][j] = 1
                        else:
                            adjacency_mat[j][i] = 0
        temp = adjacency_mat[node,...].copy()
        adjacency_mat[node,...] = adjacency_mat[...,node]
        adjacency_mat[...,node] = temp
        return adjacency_mat

    def fitness(self, state):
        state = state.reshape(self._num_nodes, self._num_nodes)
        # There's a faster way...
        rows, cols = np.where(state == 1)
        edges = zip(rows.tolist(), cols.tolist())
        gr = nx.DiGraph()
        gr.add_edges_from(edges)
        return -len([*nx.simple_cycles(gr)])

    def act(self, state, action: int) -> Tuple[List, float]:
        state = state.reshape(self._num_nodes, self._num_nodes)
        next_state = self._mutate_mat(state, action).reshape(-1)
        fitness = self.fitness(next_state)
        return next_state, (fitness, fitness == 0)

    @property
    def num_actions(self):
        return self._num_nodes
