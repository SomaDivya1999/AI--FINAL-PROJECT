import torch
import numpy as np
from collections import deque
from game import Snake, Dir
from model import Linear_QNet, QTrainer
from settings import *

class Agent:
    def __init__(self, game, pars=dict()):
        self.n_games = 0
        self.epsilon = pars.get('eps', EPSILON)
        self.eps = pars.get('eps', EPSILON)
        self.gamma = pars.get('gamma', GAMMA) # discount rate
        self.eps_range = pars.get('eps_range', EPS_RANGE)
        print(self.epsilon ,self.eps)
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(len(game.get_state()), pars.get('hidden_size', HIDDEN_SIZE), OUTPUT_SIZE)
        self.trainer = QTrainer(self.model, lr=pars.get('lr',LR), gamma=self.gamma)

        self.game = game


    
    def train_short_memory(self, *args):
        state, action, reward, next_state, done = args
        self.trainer.train_step(state, action, reward, next_state, done)

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) 
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
    
    def remember(self, *args):
        state, action, reward, next_state, done = args
        self.memory.append((state, action, reward, next_state, done))


    def get_action(self, state):
        self.epsilon = self.eps - self.n_games
        final_move = [0,0,0]
        if is_random_move(self.epsilon, self.eps_range):
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move