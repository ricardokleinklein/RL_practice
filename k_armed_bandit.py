# encoding: utf-8

import numpy as np
import matplotlib.pyplot as plt

class Bandit:
	"""Basic instance of a  series of bandits for a k-armed bandit problem.
		Args:
			(array double) bandit_probs: array of bandit success probabilities (0.0 1.0]
	"""
	def __init__(self, bandit_probs):
		self.N = len(bandit_probs)
		self.prob = bandit_probs

	def get_reward(self, action):
		"""Return reward for the action chosen. +1 if success, 0 otherwise."""
		prob = np.random.random()
		reward = 1 if (prob < self.prob[action]) else 0
		return reward


class Agent:
	"""Interface for an agent operating in k-armed bandit environment.
		Args:
			(Bandit) bandit: bandits problem environment
			(double) eps: greediness probability
	"""
	def __init__(self, bandit, eps):
		self.eps = eps
		self.Q = np.zeros(bandit.N, dtype=np.float)
		self.N = np.zeros(bandit.N, dtype=np.int)

	def update(self, action, reward):
		"""Update the number of times an action is taken and the estimation of total reward.
				Update rule given by:
					Q(a) <-- Q(a) + 1/N(a) * (R(a) - Q(a))
		"""
		self.N[action] += 1
		self.Q[action] += 1.0/self.N[action] * (reward - self.Q[action])

	def action(self, bandit):
		"""Choose an action among the bandits available using eps-greedy search."""
		prob = np.random.random()
		if prob < self.eps:
			action_explore = np.random.randint(bandit.N)
			return action_explore
		else:
			action_exploit = np.random.choice(np.flatnonzero(self.Q == self.Q.max()))
			return action_exploit


class Experiment:
	"""Simulation of N simulations of k-armed bandit problem.
	"""
	def __init__(self, bandit_probs, N_exps, N_episodes, epsilon):
		self.bandit_probs = bandit_probs
		self.N_experiments = N_exps
		self.N_episodes = N_episodes
		self.epsilon = epsilon

	def __call__(self):

		k = len(self.bandit_probs)
		print('Experiments on %d-armed bandit problem with %.3f-greedy search.' % (k, self.epsilon))

		actions_sum = np.zeros((self.N_episodes, k)) 
		rewards_avg = np.zeros(self.N_episodes)

		for exp in range(self.N_experiments):
			bandit = Bandit(self.bandit_probs)
			agent = Agent(bandit, self.epsilon)
			actions, rewards = self.run_simulation(agent, bandit)

			if (exp + 1) % (self.N_experiments / 20) == 0:
				print('[Trial %d/%d] avg_reward: %0.3f' %
					(exp + 1, self.N_experiments, np.sum(rewards)/len(rewards)))

			rewards_avg += rewards
			for j, (a) in enumerate(actions):
				actions_sum[j, a] += 1

		rewards_avg /= self.N_experiments
		print('Bandit probabilities = {}'.format(self.bandit_probs))
		return actions_sum, rewards_avg

	def run_simulation(self, agent, bandit):
		actions = []
		rewards = []
		for step in range(self.N_episodes):
			action = agent.action(bandit)
			reward = bandit.get_reward(action)
			agent.update(action, reward)
			actions.append(action)
			rewards.append(reward)
		return np.array(actions), np.array(rewards)

def plot_reward(reward, experiments, episodes, epsilon):
	plt.plot(reward)
	plt.ylabel("Rewards collected".format(experiments))
	plt.title("Bandit reward history - {} experiments w/ epsilon = {}".format(
		experiments, epsilon))
	ax = plt.gca()
	ax.set_xscale("log", nonposx='clip')
	plt.xlim([1, episodes])
	plt.xticks(fontsize=8)

def plot_actions(K, actions, experiments, episodes, epsilon):
	for i in range(K):
		action_history_sum_plot = 100 * actions[:,i] / experiments
		plt.plot(list(np.array(range(len(action_history_sum_plot)))+1), action_history_sum_plot,
			linewidth=3.0, label="Bandit #{}".format(i+1))
	plt.title("Bandit action history - {} experiments w/ epsilon = {})".format(
		experiments, epsilon), fontsize=11)
	plt.xlabel("Episode", fontsize=10)
	plt.ylabel("Bandit Action Choices (%)", fontsize=10)
	leg = plt.legend(loc='upper left', shadow=True, fontsize=6)
	ax = plt.gca()
	ax.set_xscale("log", nonposx='clip')
	plt.xlim([1, episodes])
	plt.ylim([0, 100])
	plt.xticks(fontsize=8)
	plt.yticks(fontsize=8)


def main():
	K = 10
	BANDIT_PROBS = np.random.random(K)
	N_EXPERIMENTS = 500
	EPISODES = 10000
	EPSILON = 0.1

	EXP = Experiment(BANDIT_PROBS, N_EXPERIMENTS, EPISODES, EPSILON)
	actions, rewards_avg = EXP()
	
	plt.subplot(2,1,1)
	plot_reward(rewards_avg, N_EXPERIMENTS, EPISODES,EPSILON)
	plt.subplot(2,1,2)
	plot_actions(K, actions, N_EXPERIMENTS, EPISODES, EPSILON)
	plt.show()

if __name__ == '__main__':
	main()