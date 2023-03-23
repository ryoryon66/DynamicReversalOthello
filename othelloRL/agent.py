from .othelloenv import OthelloEnv
from abc import ABC, abstractmethod

import random
import torch.nn as nn
import torch

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from collections import deque




# use abstract class to define the interface

class Agent(ABC):
        
    def __init__(self,env : OthelloEnv):
        self.env = env
        return
    
    @abstractmethod
    def get_action(self,state):
        pass

class RandomAgent():
    
    def __init__(self,env : OthelloEnv):
        self.env = env
        return
    
    def get_action(self,state):
        valid_actions = self.env.get_valid_actions()
        return random.choice(valid_actions)
    
    




class DQN(nn.Module):
    def __init__(self, input_dim = 8 * 8, num_actions : int = 8 * 8):
        super(DQN, self).__init__()


        self.layers1 = nn.Sequential(
            nn.Linear(np.prod(input_dim), 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, input_dim),
            nn.LeakyReLU()
        )
        
        self.layers2 = nn.Sequential(
            nn.Linear(input_dim, 256),

            nn.LeakyReLU(),
            nn.Linear(256, 512),
            
            nn.LeakyReLU(),
            nn.Linear(512, num_actions)
        )
        
        self.init_weights()
    
    def init_weights(self):
        for m in self.layers1:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.01)
                
        for m in self.layers2:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.01)
        
        return

    def init_last_linear(self):
        nn.init.xavier_uniform_(self.layers2[-1].weight)
        nn.init.constant_(self.layers2[-1].bias, 0.01)
        
        nn.init.xavier_uniform_(self.layers1[0].weight)
        nn.init.constant_(self.layers1[0].bias, 0.01)
        return

    def forward(self, x:torch.Tensor):
        
        assert len(x.shape) == 3
        
        x -= 1.0
        
        x /= 3.0
        
        x = x.reshape(x.size(0), -1)
        
        h = self.layers1(x)
        
        x = h 
        
        x = self.layers2(x)
        
        
        
        x = x - x.mean(dim=1,keepdim=True)
        return x





class DDQN_Agent(Agent):
    def __init__(self, path : str,env : OthelloEnv):

       
        self.input_dim = 8 * 8
        self.num_actions = 8 * 8
        
        self.env = env

        self.model = DQN(self.input_dim, self.num_actions)
        
        self.model.load_state_dict(torch.load(path))


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        return
    

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        return


    
    def _encode_action(self,action : tuple[int,int]) -> int:
        
        return action[0] * self.env.board_size + action[1]
    
    def _decode_action(self,action : int) -> tuple [int,int]:
        
        return action // self.env.board_size, action % self.env.board_size

        
    

    def get_action(self, state : np.ndarray, epsilon : float =  0):
        """tupleで表される行動を返す

        Args:
            state (_type_): _description_
            epsilon (float, optional): _description_. Defaults to 0.1.

        Returns:
            _type_: _description_
        """
        
        if epsilon is None:
            epsilon = self.epsilon
        
        assert len(state.shape) == 2
        
        if np.random.rand() <= epsilon:
            action = random.choice(self.env.get_valid_actions())
            return action
        state = Variable(torch.from_numpy(state).unsqueeze(0).float()).to(self.device)
        q_values = self.model(state)
        
        # 合法手から最大のQ値を持つ行動を選択
        valid_actions = self.env.get_valid_actions()
        valid_actions = [self._encode_action(action) for action in valid_actions]
        q_values = q_values[0,valid_actions]
        action = valid_actions[torch.argmax(q_values).item()]
        decoded_action = self._decode_action(action)
        return decoded_action


if __name__ == "__main__":
    env = OthelloEnv()
    agent = DDQN_Agent("ddqn_best_weights.pth",env)
    state = env.reset()

    # 先手　DDQN 後手 人間
    
    env.render()
    done = False
    
    print("BLACK : DDQN","WHITE : HUMAN",sep = " ")
    
    while True:
        
        assert (env.get_state() == agent.env.get_state()).all()
        
        while env.player == env.MAIN_PLAYER:
            valid_actions = env.get_valid_actions()
            print ("valid actions DDQN : ",valid_actions)
            evaluation = agent.model(Variable(torch.from_numpy(env.get_state()).unsqueeze(0).float()).to(agent.device))
            valid_actions_encoded = [agent._encode_action(action) for action in valid_actions]
            evaluation = evaluation[0,valid_actions_encoded]
            print("evaluation : ",evaluation.tolist())
            action = agent.get_action(state)
            state,reward,done,info = env.step(action,env.MAIN_PLAYER)
            env.render()
            if done:
                break
        
        if done:
            break
        
        while env.player == env.OPPONENT_PLAYER:
            valid_actions = env.get_valid_actions()
            print("valid actions : ",valid_actions)
            action = tuple(map(int,input().split()))
            if action not in valid_actions:
                print("invalid action")
                continue
            state,reward,done,info = env.step(action,env.OPPONENT_PLAYER)
            env.render()
            if done:
                break
        
        if done:
            break
    
    print (env.scores)
    
        