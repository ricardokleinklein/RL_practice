"""
	Estimators for K-Armed-like problems
"""

import numpy as np


class Agent:
	"""
		Base class for estimator agents in K-Armed Bandit-like problems

		Args:
			init_action_value: (K,) np.array initial action values
	"""

	def __init__(self, init_action_value):
		self.action_value_hat = init_action_value
		self.k = len(init_action_value)
		self.action_selection = np.zeros(self.k, dtype="int64")

	def select_action(self):
		raise NotImplementedError(
			"Need to implement action selection")

	def update(self):
		raise NotImplementedError(
			"Need to implement update of estimates")

	def select_greedy_action(self):
		return np.argmax(self.action_value_hat)

	def select_random_action(self):
		return np.random.choice(self.k)


class SampleAverageAgent(Agent):
	"""
		Agent whose estimations are based on sample average
	"""

	def __init__(self, init_action_value, eps):
		super(SampleAverageAgent, self).__init__(init_action_value)
		self.eps = eps

	def select_action(self):
		probability = np.random.rand()
		if probability >= self.eps:
			return self.select_greedy_action()

		return self.select_random_action()

	def update(self, action, reward):
		self.action_selection[action] += 1
		q_n = self.action_value_hat[action]
		n = self.action_selection[action]
		self.action_value_hat[action] = q_n +  (1. / n) * (reward - q_n)


class WeightedAverageAgent(SampleAverageAgent):
	"""
		Weighted-average estimator
	"""
	def __init__(self, init_action_value, eps, alpha=0.1):
		super(WeightedAverageAgent, self).__init__(init_action_value, eps)
		self.alpha = alpha

	def update(self, action, reward):
		self.action_selection[action] += 1	
		q_n = self.action_value_hat[action]
		self.action_value_hat[action] = q_n + self.alpha * (reward - q_n)


class UCBAgent(WeightedAverageAgent):
	"""
		Upper-Confidence-Bound Action Selection
	"""
	def __init__(self, init_action_value, eps, alpha=0.1, c=1):
		super(UCBAgent, self).__init__(init_action_value, eps, alpha)
		self.c = c
		self.t = 0

	def select_action(self):
		self.t += 1
		probability = np.random.rand()
		if probability >= self.eps:
			return self.select_greedy_action()

		return self.select_ucb_action()

	def estimate_action_potential(self, action):
		q_t = self.action_value_hat[action]
		ln_t = np.log(self.t)
		n_t = self.action_selection[action]

		return q_t + self.c * np.sqrt(ln_t / n_t)

	def select_ucb_action(self):
		greedy_action = {self.select_greedy_action()}
		if 0 in self.action_selection:
			actions_unselected = [a for a in range(self.k) if 
				self.action_selection[a] == 0]
			selected = np.random.choice(actions_unselected)
			self.action_selection[selected] += 1
			return selected

		non_greedy_action = set(range(self.k)) - greedy_action
		action_potential = [self.estimate_action_potential(a) for a 
			in non_greedy_action]
		return np.argmax(non_greedy_action)








