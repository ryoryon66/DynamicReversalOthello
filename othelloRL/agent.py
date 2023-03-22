from othelloenv import OthelloEnv
from abc import ABC, abstractmethod
import random

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