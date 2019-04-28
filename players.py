import abc

import numpy as np
import time


class Player:

	def __init__(self, number):
		self.number = number

	@abc.abstractmethod
	def make_move(self, board):
		pass

	@abc.abstractmethod
	def feedback(self, points):
		pass


class Human(Player):

	def feedback(self, points):
		pass

	def __init__(self, number):
		super().__init__(number)

	def make_move(self, board):
		move = None

		while True:
			choice = input(f"Player {self.number} make a move (letter|number): ")
			if len(choice) == 2 and choice[0] in 'abcABC' and choice[1] in '123':
				move = [None, None]
				move[1] = ord(choice[0].lower()) - ord('a')
				move[0] = int(choice[1]) - 1
				if board[move[0], move[1]] is None:
					break

		return move


class RandomAI(Player):

	def __init__(self, number, delay=.5):
		super().__init__(number)
		self.delay = delay

	def make_move(self, board):
		time.sleep(self.delay)
		move = np.random.choice(3, 2)
		while board[move[0], move[1]] is not None:
			move = np.random.choice(3, 2)
		return move

	def feedback(self, points):
		pass
