"""
	K-Armed Bandit Testbed class
"""

import numpy as np
from numpy.random import normal

class KArmedTestbed:
	"""
		K-Armed Bandit Testbed

		q*-values start being equal for every of the k actions and then
		each of them takes random walk

		Args:
			k (int): Number of actions
	"""

	def __init__(self, k, stddev=0.01):
		self.k = k
		self.stddev = stddev
		self.q_star = np.zeros(k)

	def __getitem__(self, i):
		return self.q_star[i]

	def random_walk(self):
		increment = normal(loc=0, scale=self.stddev,
			size=self.k)
		self.q_star += increment

	def fixed_increment(self):
		"""
			One option is progressively highlighted as more rewarding.
		"""
		# TODO
		raise NotImplementedError("Not implemented")

	def get_optimal_action(self):
		return np.argmax(self.q_star)

	def get_optimal_action_value(self):
		return np.max(self.q_star)

	def is_optimal_action(self, action):
		return float(self.get_optimal_action() ==  action)

	def sample_reward(self, action):
		return normal(loc=self.q_star[action], scale=1)