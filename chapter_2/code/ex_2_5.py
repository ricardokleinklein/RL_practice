"""
	Exercise 2.5 from Reinforcement Learning: An Introduction
	2nd edition by Richard S. Sutton and Andrew G. Barto
"""
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from agents import *
from testbed import KArmedTestbed


np.random.seed(250)


def plot_performance(legend, reward, optim_action):
	"""
		Plot the performance of a series of experiments.
		Namely, the average reward by timestep and the relative 
		amount of times that the chosen action was the optimal one

		Args:
			legend: (N,) Name for every agent experimented on
			reward: (N, N_RUNS, N_STEPS) Rewards
			optim_action: (N, N_RUNS, N_STEPS) Whether optimal or not 
	"""
	for i, agent in enumerate(legend):
		avg_reward = np.average(reward[i], axis=0)
		plt.plot(avg_reward, label=agent)

	plt.subplot(2, 1, 1)
	plt.legend()
	plt.xlabel('Steps')
	plt.ylabel('Average reward')

	for i, agent in enumerate(legend):
		average_optimality = np.average(optim_action[i], axis=0)
		plt.plot(average_optimality, label=agent)

	plt.subplot(2, 1, 2)
	plt.legend()
	plt.xlabel('Steps')
	plt.ylabel('Optimal action')
	plt.show()


if __name__ == "__main__":
	# Experiment setup:
	N_ESTIMATORS = 3
	K = 10
	N_STEPS = 10000
	N_RUNS = 2000
	EPS = 0.1
	ALPHA = 0.1

	# Results table init:
	reward = np.zeros((N_ESTIMATORS, N_RUNS, N_STEPS))
	optim_action = np.zeros((N_ESTIMATORS, N_RUNS, N_STEPS))

	# Experiment!
	for run in tqdm(range(N_RUNS)):
		testbed = KArmedTestbed(K)

		init_action_value = np.zeros(K)
		agent1 = SampleAverageAgent(
			init_action_value.copy(), eps=EPS)
		agent2 = WeightedAverageAgent(
			init_action_value.copy(), eps=EPS, alpha=ALPHA)
		agent3 = UCBAgent(init_action_value.copy(), eps=EPS, alpha=ALPHA)
		agents = [agent1, agent2, agent3]

		for step in range(N_STEPS):
			for j, agent in enumerate(agents):
				action = agent.select_action()
				is_optimal = testbed.is_optimal_action(action)
			
				r = testbed.sample_reward(action)

				agent.update(action, r)

				reward[j, run, step] = r
				optim_action[j, run, step] = is_optimal

				testbed.random_walk()

	plot_performance(["Sample-Averaged", "Weighted-Average", "UCB"], 
		reward, optim_action)




	
