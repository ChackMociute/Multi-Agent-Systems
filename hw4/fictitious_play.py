import numpy as np
from random import choices


class Agent:
    def __init__(self, actions, rewards, probs=None):
        self.best_response = False
        self.actions = self.create_actions(actions, probs)
        self.rewards = rewards
    
    @staticmethod
    def create_actions(actions, probs):
        if probs is None:
            return {a: 1/len(actions) for a in actions}
        if len(actions) != len(probs):
            raise RuntimeError("Actions and probabilities should have the same length")
        if sum(probs) != 1:
            raise ValueError("Probabilities should add up to 1")
        return {a: p for a, p in zip(actions, probs)}
    
    def set_opp_count(self):
        self.opp_count = {a: 0 for a in self.rewards[self.select_action()].keys()}
        
    def record_opponent_action(self, opp_action):
        if opp_action not in self.opp_count.keys():
            raise KeyError("Opponent action different from the list of possible actions")
        self.opp_count[opp_action] += 1
    
    def select_action(self):
        if self.best_response:
            return self.select_best_response()
        return choices(list(self.actions.keys()), weights=self.actions.values())[0]
    
    def select_best_response(self):
        s = sum(self.opp_count.values())
        utilities = {a: sum([self.rewards[a][a2] * c / s for a2, c in self.opp_count.items()])
                     for a in self.actions.keys()}
        return max(utilities, key=utilities.get)


class Game:
    def __init__(self, agents, n_initial_games=1, total_games=10000):
        self.n_initial_games = n_initial_games
        self.total_games = total_games
        self.agents = agents
    
    def play_game(self):
        a1, a2 = self.agents
        a1.record_opponent_action(a2.select_action())
        a2.record_opponent_action(a1.select_action())
    
    def play(self):
        for agent in self.agents:
            agent.set_opp_count()
            agent.best_response = False
        # Play some initial games randomly to accrue some data
        for _ in range(self.n_initial_games):
            self.play_game()
        # Switch to playing best responses
        for agent in self.agents:
            agent.best_response = True
        for _ in range(self.total_games - self.n_initial_games):
            self.play_game()
    
    def summary(self):
        print(f"Player 1 plays the actions {tuple(game.agents[1].opp_count.keys())} with probabilities " +\
              f"{tuple(round(c / self.total_games, 3) for c in game.agents[1].opp_count.values())}")
        print(f"Player 2 plays the actions {tuple(game.agents[0].opp_count.keys())} with probabilities " +\
              f"{tuple(round(c / self.total_games, 3) for c in game.agents[0].opp_count.values())}")

        
        
A1, A2 = ['A', 'B', 'C'], ['W', 'X', 'Y', 'Z']
rewards = np.asarray([
    [[1,5], [2,2], [3,4], [3,1]],
    [[3,0], [4,1], [2,5], [4,2]],
    [[1,3], [2,6], [5,2], [2,3]]
])
R1 = {a1: {a2: r[0] for a2, r in zip(A2, R)} for a1, R in zip(A1, rewards)}
R2 = {a1: {a2: r[1] for a2, r in zip(A1, R)} for a1, R in zip(A2, rewards.swapaxes(0, 1))}
game = Game([Agent(A1, R1), Agent(A2, R2)])

game.play()
game.summary()