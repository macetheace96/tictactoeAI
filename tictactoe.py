import os
import sys

import numpy as np

from players import *
from genetic_ai import *


WIN = 10
LOSS = -10
DRAW = 0
INVALID = -1


class TicTacToe:

	def __init__(self):

		self.player1 = None
		self.player2 = None

		self.board = np.array([[None, None, None], [None, None, None], [None, None, None]])

		self.curr_player = 1

	def get_move(self):
		player = self.get_curr_player()
		move = player.make_move(self.board)
		return move

	def update_board(self, move):
		self.board[move[0], move[1]] = self.curr_player

	def game_over(self):
		result = None

		if any(all(row == 1) or all(row == 2) for row in self.board):
			result = WIN
		elif any(all(col == 1) or all(col == 2) for col in self.board.transpose()):
			result = WIN
		elif all(self.board.diagonal() == 1) or all(self.board.diagonal() == 2):
			result = WIN
		elif all(self.board[::-1].diagonal() == 1) or all(self.board[::-1].diagonal() == 2):
			result = WIN
		elif all(self.board.flatten() != None):
			result = DRAW
		else:
			result = None

		return result

	def restart_game(self):
		self.board = np.array([[None, None, None], [None, None, None], [None, None, None]])
		self.curr_player = 1

	def get_curr_player(self):
		if self.curr_player == 1:
			return self.player1
		else:
			return self.player2

	def get_next_player(self):
		if self.curr_player == 1:
			return self.player2
		else:
			return self.player1

	def get_players(self):
		player1, player2 = None, None
		num = 1
		while not player1:
			player1 = input("Player 1: 'human' or 'ai' [genetic, random]? ")
			if player1 == 'human':
				self.player1 = Human(1)
			elif player1 == 'ai' or player1 == 'genetic':
				self.player1 = GeneticAI(1, delay=.5)
			elif player1 == 'random':
				self.player1 = RandomAI(1, delay=.5)
			else:
				player1 = None
				print("Player 1 must be either human or one of the ai.")

		while not player2:
			player2 = input("Player 2: 'human' or 'ai' (genetic, random)? ")
			if player2 == 'human':
				self.player2 = Human(2)
			elif player2 == 'ai' or player2 == 'genetic':
				self.player2 = GeneticAI(2, delay=.5)
			elif player2 == 'random':
				self.player2 = RandomAI(2, delay=.5)
			else:
				print("Player 2 must be either human or one of the ai.")
				player2 = None

	def show_board(self):
		os.system('clear')
		s = "  A B C\n -------\n"
		for i, row in enumerate(self.board):
			s += f"{i+1}|"
			for j, cell in enumerate(row):
				if cell == 1:
					s += "X"
				elif cell == 2:
					s += "O"
				else:
					s += " "
				s += "|"
				if j == 2:
					s += "\n"
			s += " -------\n"
		print(s)

	def training_run(self, player1, player2, n_games, verbose=3):
		self.player1 = player1
		self.player2 = player2

		for i in range(n_games):
			if verbose >= 1:
				print(f"GAME {i+1}")
			self.restart_game()
			while True:
				self.update_board(self.get_move())
				if verbose >= 3:
					self.show_board()
				result = self.game_over()
				if result is not None:
					if result == WIN:
						if verbose >= 2:
							print(f"PLAYER {self.curr_player} WINS")
						self.get_curr_player().feedback(WIN)
						self.get_next_player().feedback(LOSS)
					elif result == DRAW:
						if verbose >= 2:
							print(f"IT'S A DRAW")
						self.get_curr_player().feedback(DRAW)
						self.get_next_player().feedback(DRAW)
					break
				else:
					self.curr_player = 1 if self.curr_player == 2 else 2

	def run(self):
		print(f"TIC TAC TOE")
		self.get_players()

		while True:
			self.show_board()
			move = self.get_move()
			self.update_board(move)
			self.show_board()
			result = self.game_over()
			if result is not None:
				if result == WIN:
					print(f"PLAYER {self.curr_player} WINS")
				elif result == DRAW:
					print(f"IT'S A DRAW")
				restart = input("Play again (ENTER|n)? ")
				if restart in ['n', 'no']:
					sys.exit()
				else:
					self.restart_game()
			else:
				self.curr_player = 1 if self.curr_player == 2 else 2


if __name__ == "__main__":
	ttt = TicTacToe()
	ttt.run()
