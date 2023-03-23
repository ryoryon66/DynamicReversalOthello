import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from collections import deque
from othelloenv import OthelloEnv
from agent import Agent,RandomAgent
from env_wrapper import WrappedEnv
import matplotlib.pyplot as plt
import pandas as pd


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


class DDQN_Learner(Agent):
    def __init__(self, env : WrappedEnv):
        self.env = env
        self.gamma = 0.999999

        self.batch_size = 128
        self.memory = deque(maxlen=400000) #  (state, action, reward, next_state, done)
        
        self.input_dim = 8 * 8
        self.num_actions = 8 * 8

        self.model = DQN(self.input_dim, self.num_actions)
        self.target_model = DQN(self.input_dim, self.num_actions)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.000001,weight_decay=0.01)

        self.update_target_model()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.target_model.to(self.device)
        
        self.epsilon = 1.0
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.01
        
        
        
        return
    

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        return

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        return
    
    def _encode_action(self,action : tuple[int,int]) -> int:
        
        return action[0] * self.env.board_size + action[1]
    
    def _decode_action(self,action : int) -> tuple [int,int]:
        
        return action // self.env.board_size, action % self.env.board_size

        
    

    def get_action(self, state : np.ndarray, epsilon : float = None):
        """0-63で表される行動を返す

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

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        
        action_batch = [self._encode_action(action) for action in action_batch]

        state_batch = Variable(torch.from_numpy(np.stack(state_batch)).float()).to(self.device)
        action_batch = Variable(torch.from_numpy(np.array(action_batch))).to(self.device)
        reward_batch = Variable(torch.from_numpy(np.array(reward_batch)).float()).to(self.device)
        next_state_batch = Variable(torch.from_numpy(np.stack(next_state_batch)).float()).to(self.device)
        


        q_values = self.model(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_state_batch).max(1)[0]

        expected_q_values = reward_batch + self.gamma * next_q_values * (1 - torch.tensor(done_batch).float()).to(self.device)
        
        # diff = expected_q_values - q_values debug print the diff

        print ("expected_q_values and q_values diff")
        print ((expected_q_values - q_values).abs().mean().item())

        loss = nn.HuberLoss()(q_values, expected_q_values.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        
        return loss.item()



    def train(self, episodes : int):
        
        sum_reward = 0.0
        
        
        learner_scores = []
        opponent_scores = []
        
        win_rate = []
        
        rewards = []
        
        loss_history = []
        
        best_agent = (0, -np.inf) # (episode, average_win_rate)

        for episode in range(episodes):
            state = self.env.reset(player=1)
            done = False
            steps = 0
            
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                
            

            while not done:
                
                
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)


                self.remember(state, action, reward, next_state, done)
                state = next_state
                steps += 1

                if done:
                    
                    learner_scores.append(self.env.scores[1])
                    opponent_scores.append(self.env.scores[2])
                    rewards.append(reward)
                    print("Episode: {}/{}, Steps: {}, Epsilon: {:.5}".format(episode + 1, episodes, steps, self.epsilon))
                    print ("best agent",best_agent)

            loss = self.replay()
            loss_history.append(loss)
            
            sum_reward += reward
            
            
            # 一定確率でイプシロンをもどす
            if  np.random.rand() < 0.003:
                self.epsilon = min(1.0, self.epsilon + 0.02)
            
            # print("score",self.env.scores)
            
            
            if episode % 100 == 0:
                print(f'episode:{episode}, ave_reward:{sum_reward / 100.0}')
                sum_reward = 0.0
                self.update_target_model()
            
            
            if episode % 500 == 0:
                torch.save(self.model.state_dict(), f'ddqn_weights{episode}.pth')
                
                # visualize the score and save the figure
                
                plt.plot(learner_scores, label='learner')
                plt.plot(opponent_scores, label='opponent')
                plt.legend()
                plt.savefig(f'./ddqn_scores.png')
                plt.close()
                
                # visualize the diffence between learner and opponent
                
                diff = np.array(learner_scores) - np.array(opponent_scores)
                plt.plot(diff, label='diff')
                # 移動平均
                diff_ma = pd.Series(diff).rolling(100).mean()
                plt.plot(diff_ma, label='diff_ma')
                plt.legend()
                plt.savefig(f'./ddqn_diff.png')
                plt.close()
                
                # rewardは移動平均をとる
                rewards_ma = pd.Series(rewards).rolling(100).mean()
                plt.plot(rewards_ma, label='reward')
                plt.legend()
                plt.savefig(f'./ddqn_rewards.png')
                plt.close()
                
                # loss
                loss_ma = pd.Series(loss_history).rolling(1).mean()
                plt.plot(loss_ma, label='loss')
                plt.legend()
                plt.savefig(f'./ddqn_loss.png')
                plt.close()
            
            if episode % 500 == 0: # 500エピソードごとに勝率を計算
                win_rate.append(self.evaluate_agent(200))
                xasix = np.arange(0,episode+1,500)
                plt.plot(xasix,win_rate, label='win_rate')
                plt.legend()
                plt.savefig(f'./ddqn_win_rate.png')
                plt.close()
                
                # 最高のエージェントを保存
                if win_rate[-1] > best_agent[1]:
                    best_agent = (episode, win_rate[-1])
                    
                    torch.save(self.model.state_dict(), 'ddqn_best_weights.pth')
                    
                    print ("best agent is updated")
                    print (f"episode:{best_agent[0]}, win_rate:{best_agent[1]}")
            
            if episode % 3000 == 0: # 2000エピソードごとにlast layerを初期化
                
                self.model.init_last_linear()
                self.update_target_model()
            
            


        torch.save(self.model.state_dict(), 'ddqn_weights.pth')
        
    
    def evaluate_agent(self,n_game : int = 100):
        
        numwin = 0
        
        for i in range(n_game):
            
            state = self.env.reset(player=1)
            done = False
            
            while not done:
                action = self.get_action(state,epsilon=0.0)
                next_state, reward, done, _ = self.env.step(action)
                state = next_state
                
            if reward > 0.5:
                numwin += 1
            
        
        return numwin / n_game

    def load_weights(self, path):
        self.model.load_state_dict(torch.load(path))
        self.update_target_model()
        return

if __name__ == "__main__":
    env = OthelloEnv()
    opponent_agent = RandomAgent(env)
    wrapped_env = WrappedEnv(env, opponent_agent)
    agent = DDQN_Learner(wrapped_env)
    agent.train(1000000)