"""
Template for implementing QLearner  (c) 2015 Tucker Balch
"""

import numpy as np
import random as rand
from collections import deque

class QLearner(object):

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):

        self.verbose = verbose
        self.num_actions = num_actions
        self.num_states = num_states
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.s = 0
        self.a = 0
        self.Q = np.zeros((num_states, num_actions))

        self.replay_mem = deque(maxlen=self.dyna + 50)
    
    def get_action(self, s):
        """
        @summary: Get next action based on the specified state
        @param s: state
        @returns: action
        """
        p = rand.random()
        if p < self.rar:
            return rand.randint(0, self.num_actions - 1)
        return np.argmax(self.Q[s])
    
    def update_Q(self, s, a, s_prime, r):
        prev_r = self.Q[s, a]
        later_r = np.max(self.Q[s_prime])
        updated_r = (1 - self.alpha) * prev_r + self.alpha * (r + later_r)
        self.Q[s, a] = updated_r
    
    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        self.a = self.get_action(s)
        return self.a

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """
        self.update_Q(self.s, self.a, s_prime, r)
        if self.dyna > 0:
            self.run_dyna(self.s, self.a, s_prime, r)
        next_a = self.get_action(s_prime)
        self.s = s_prime
        self.a = next_a
        self.rar = self.rar * self.radr
        return self.a
    
    def run_dyna(self, s, a, s_prime, r):
        self.replay_mem.append((s, a, s_prime, r))
        for i in range(self.dyna):
            self.hallucinate()
    
    def hallucinate(self):
        experience = rand.randint(0, len(self.replay_mem) - 1)
        s, a, s_prime, r = self.replay_mem[experience]
        self.update_Q(s, a, s_prime, r)

    def author(self):
        return "chabinshuti3"