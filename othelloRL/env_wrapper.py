from .othelloenv import OthelloEnv
from .agent import Agent

class WrappedEnv():
    
    def __init__(self, env : OthelloEnv,opponent_agent : Agent):
        
        self.env = env
        self.opponent_agent = opponent_agent
        self.board_size = env.board_size
        self.board = env.board
        self.scores = env.scores
        self.MAIN_PLAYER = env.MAIN_PLAYER
        self.OPPONENT_PLAYER = env.OPPONENT_PLAYER
        
        
        
        return
    
    def step(self,action : tuple[int,int]):
        """
        学習エージェントの行動を受け取り、環境の更新を行う。
        stepした後はまた学習エージェントのターンとなる。

        Args:
            action (tuple[int,int]): どのマスに石を置くかを表すタプル
        
        Returns:
            state (np.ndarray): 環境の状態
            reward (int): 報酬
            done (bool): ゲーム終了フラグ
            info (dict): ゲーム終了の原因を示す辞書
        """
        
        assert self.env.player == self.MAIN_PLAYER
        
        assert (self.env.is_valid_move(action[0],action[1]),"invalid action")
        
        
        
        # take action
        
        state,reward,done,info = self.env.step(action,self.MAIN_PLAYER)
        
        if done:
            return state,reward,done,info
        
        while self.env.player != self.MAIN_PLAYER:
            opponent_action = self.opponent_agent.get_action(self.env.get_state())
            state,reward,done,info = self.env.step(opponent_action,self.OPPONENT_PLAYER)
            
            if done:
                return state,reward,done,info
        
        return state,reward,done,info
        

    def reset(self,player):
        self.env.reset()
        self.env.player = player
        self.player = player
        self.scores = self.env.scores
        return self.env.get_state()

    def get_state(self):
        return self.env.get_state()
    
    def get_valid_actions(self) -> list[tuple]:
        return self.env.get_valid_actions()
    
    def is_valid_move(self,x,y):
        return self.env.is_valid_move(x,y)
    
    