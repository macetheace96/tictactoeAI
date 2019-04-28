import os
import time
from copy import deepcopy
from datetime import datetime

import torch
from torch import nn
import numpy as np

from players import Player
from tictactoe import *


class GeneticAI(Player):

	def __init__(self, number, net=None, delay=0):
		super().__init__(number)

		self.input_dim = 10
		self.hidden_dim = 20
		self.output_dim = 9

		self.net = net
		if self.net is None:
			self.load_net()

		self.delay = delay

		self.score = 0

	def train_genetic(self, pop_size=20, n_gens=1000, sigma=.1, resume_dir=None, save_dir=None):
		if not save_dir:
			if not resume_dir:
				save_dir = f"weights/genetic/{datetime.now()}"
				os.mkdir(save_dir)
			else:
				save_dir = resume_dir
		population = self.init_population(pop_size, resume_dir)
		population = self.evaluate(population)

		for i in range(n_gens):
			print(f"GENERATION {i}")
			parents = self.select(population)
			population = self.cross(population, parents)
			population = self.mutate(population, sigma)
			population = self.evaluate(population)
			self.save_net(population[0], save_dir)
			population = self.cull(population, pop_size)
			self.save_population(population, save_dir)

		self.load_net()

	def init_population(self, pop_size, resume_dir, sigma=5):
		pop = []
		for i in range(pop_size):
			net = nn.Sequential(
				nn.Linear(self.input_dim, self.hidden_dim),
				nn.ReLU(),
				nn.Linear(self.hidden_dim, self.hidden_dim),
				nn.ReLU(),
				nn.Linear(self.hidden_dim, self.output_dim),
				nn.Softmax()
			)
			if resume_dir:
				net.load_state_dict(torch.load(os.path.join(resume_dir, f"{i}.pt")))
			for param in net.parameters():
				param.data = torch.tensor(np.random.normal(0, sigma, param.data.shape), dtype=torch.float)
			pop.append(net)
		return pop

	def evaluate(self, population):
		if len(population) == 1:
			return population

		winners = []
		losers = []
		result = []

		game = TicTacToe()

		indices = np.random.permutation(len(population))

		pairings = [pair for pair in zip(indices[::2], indices[1::2])]
		if len(pairings) < len(population) / 2:
			if np.random.randint(2) == 0:
				winners.append(population[indices[-1]])
			else:
				winners.append(population[indices[-1]])

		for id1, id2 in pairings:
			player1 = GeneticAI(1, population[id1])
			player2 = GeneticAI(2, population[id2])

			game.training_run(player1, player2, 1, verbose=0)

			if player1.score >= player2.score:
				winners.append(population[id1])
				losers.append(population[id2])
			else:
				winners.append(population[id2])
				losers.append(population[id1])

		result.extend(self.evaluate(winners))
		result.extend(self.evaluate(losers))

		return result

	def select(self, population):
		probs = np.array([1/(i+1) for i in range(len(population))])
		parent_ids = np.random.choice(len(population), len(population)//2, replace=False, p=probs/sum(probs))
		parents = [population[i] for i in parent_ids]
		return list(zip(parents[:len(parents)//2], parents[:len(parents)//2:-1]))

	def cross(self, population, parents):
		crossed_population = population.copy()
		for parent1, parent2 in parents:
			child1 = deepcopy(parent1)
			child2 = deepcopy(parent2)
			for parent_param1, parent_param2, child_param1, child_param2 in zip(parent1.parameters(), parent2.parameters(), child1.parameters(), child2.parameters()):
				parent_params = np.random.permutation([parent_param1, parent_param2])
				child_param1.data.copy_(parent_params[0].data)
				child_param2.data.copy_(parent_params[1].data)
			crossed_population.extend([child1, child2])
		return crossed_population

	def mutate(self, population, sigma):
		mutated_pop = deepcopy(population)
		for net in mutated_pop:
			for param in net.parameters():
				param.data += torch.tensor(np.random.normal(param.data.numpy(), sigma, param.data.shape), dtype=torch.float)
		return mutated_pop

	def cull(self, population, pop_size):
		probs = np.array([1 / (i + 1) for i in range(len(population))])
		survivor_ids = np.random.choice(len(population), pop_size, replace=False, p=probs / sum(probs))
		return [population[i] for i in survivor_ids]

	def save_population(self, population, save_dir):
		for i in range(len(population)):
			torch.save(population[i].state_dict(), os.path.join(save_dir, f"{i}.pt"))

	def save_net(self, net, path="weights/genetic"):
		torch.save(net.state_dict(), os.path.join(path, "genetic_ai_weights.pt"))

	def load_net(self, path="weights/genetic"):
		self.net = nn.Sequential(
			nn.Linear(self.input_dim, self.hidden_dim),
			nn.ReLU(),
			nn.Linear(self.hidden_dim, self.hidden_dim),
			nn.ReLU(),
			nn.Linear(self.hidden_dim, self.output_dim),
			nn.Softmax()
		)
		self.net.load_state_dict(torch.load(os.path.join(path, "genetic_ai_weights.pt")))

	def make_move(self, board):
		time.sleep(self.delay)

		x = board.copy()
		x[x == None] = -1
		x = np.append(x, self.number)
		x = torch.tensor(x.astype(float), dtype=torch.float, requires_grad=False)

		pred = np.argsort(self.net(x).detach().numpy())
		for id in pred[::-1]:
			move = np.unravel_index(id, (3, 3))
			if board[move] is None:
				return move

	def feedback(self, points):
		self.score += points


if __name__ == "__main__":
	ai = GeneticAI(0)
	ai.train_genetic(resume_dir="/Users/masonfp/Desktop/cs/tictactoeAI/weights/genetic/2019-04-28 19:08:33.292615")
