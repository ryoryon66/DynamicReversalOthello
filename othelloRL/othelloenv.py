import numpy as np

class OthelloEnv:
    def __init__(self, board_size=8):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=int)
        self.player = 1
        self.reset()
        
        self.scores = {1: 0, 2: 0}
        self.MAIN_PLAYER = 1
        self.OPPONENT_PLAYER = 2

    def reset(self):
        self.board.fill(0)
        center = self.board_size // 2
        self.board[center-1:center+1, center-1:center+1] = np.array([[2, 1], [1, 2]])
        self.player = 1
        self.scores = {1: 0, 2: 0}
        return self.get_state()

    def get_state(self):
        return self.board.copy()

    def get_valid_actions(self) -> list[tuple] :
        valid_actions = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.is_valid_move(i, j):
                    valid_actions.append((i, j))
        return valid_actions

    def is_valid_move(self, x, y):
        # Check if the position is empty
        if self.board[x, y] != 0:
            return False

        # Check all directions
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for dx, dy in directions:
            if self.check_direction(x, y, dx, dy):
                return True
        return False

    def check_direction(self, x, y, dx, dy):

        nx, ny = x + dx, y + dy
        if not (0 <= nx < self.board_size and 0 <= ny < self.board_size):
            return False   # Out of board
        if self.board[nx, ny] != 3 - self.player:
            return False  # Not opponent's disc

        while 0 <= nx < self.board_size and 0 <= ny < self.board_size:
            nx, ny = nx + dx, ny + dy
            if not (0 <= nx < self.board_size and 0 <= ny < self.board_size):
                break
            if self.board[nx, ny] == self.player:
                return True
            elif self.board[nx, ny] == 0:
                break
        return False

    def step(self, action,player):
        
        """1ターン分の環境の更新を行う。

        Returns:
            state (np.ndarray): 環境の状態
            reward (int): 報酬
            done (bool): ゲーム終了フラグ
            info (dict): ゲーム終了の原因を示す辞書
        """
        
        assert(self.player == player,"player is not correct")
        
        x, y = action
        if not self.is_valid_move(x, y):
            return self.get_state(), -1, True, {"invalid_move": True}

        self.board[x, y] = self.player
        self.flip_discs(x, y)

        self.player = 3 - self.player
        
        # Check if the game is finished
        if len(self.get_valid_actions()) == 0:
            self.player = 3 - self.player
            # Check if opponent can move or not
            if len(self.get_valid_actions()) == 0:
                return self.get_state(), self.calculate_reward(), True, {"game_finished": True}
            else:
                return self.get_state(), 0, False, {"skip_turn": True}

        return self.get_state(), 0, False, {}


    def flip_discs(self, x, y):
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for dx, dy in directions:
            if self.check_direction(x, y, dx, dy):
                self.flip_discs_in_direction(x, y, dx, dy)
        
        return

    def flip_discs_in_direction(self, x, y, dx, dy):
        nx, ny = x + dx, y + dy
        while 0 <= nx < self.board_size and 0 <= ny < self.board_size:
            if self.board[nx, ny] == self.player:
                break
            elif self.board[nx, ny] == 3 - self.player:
                self.board[nx, ny] = self.player
                nx, ny = nx + dx, ny + dy
                
                self.scores[self.player] += 1
            
        

    def calculate_reward(self):
        """Main Playerにとってのゲーム終了時の報酬を計算する。

        Returns:
            float: 報酬
        """
        
        main_player_score = self.scores[self.MAIN_PLAYER]
        opponent_score = self.scores[self.OPPONENT_PLAYER]
        
        
        if main_player_score > opponent_score:
            return 1.0
        elif main_player_score < opponent_score:
            return -1.0
        else:
            return - 0.1

    def render(self):
        
        print ("scores:",self.scores)
        
        # print col numbers
        print("  0 1 2 3 4 5 6 7")
        
        for i in range(self.board_size):
            # print row numbers
            print(i, end=" ")
            for j in range(self.board_size):
                if self.board[i, j] == 1:
                    print("B", end=" ")
                elif self.board[i, j] == 2:
                    print("W", end=" ")
                else:
                    print(".", end=" ")
            print()
        print()


import random

def cpu_action(env):
    valid_actions = env.get_valid_actions()
    if valid_actions:
        return random.choice(valid_actions)
    return None

def human_action(env):
    valid_actions = env.get_valid_actions()
    print("Valid actions:", valid_actions)
    while True:
        try:
            x, y = input("Enter your move (x, y): ").split()
            x, y = int(x), int(y)
            if (x, y) in valid_actions:
                return x, y
            else:
                print("Invalid move. Please enter a valid move.")
        except Exception as e:
            print("Invalid input. Please enter the coordinates in the format x, y.")

def play_game():
    env = OthelloEnv()

    while True:
        env.render()
        print("score" , "Human:" , env.scores[1] , "CPU:" , env.scores[2])
        if env.player == 1:
            print("Player 1 (Human)")
            action = human_action(env)
            state, reward, done, info = env.step(action)
            if done:
                break
        else:
            print("Player 2 (CPU)")
            action = cpu_action(env)
            if action is not None:
                state, reward, done, info = env.step(action)
                if done:
                    break
            else:
                print("CPU has no valid moves. Skipping turn.")
                env.player = 3 - env.player

    env.render()
    if reward == 1:
        print("Congratulations, you won!")
    elif reward == -1:
        print("CPU won. Better luck next time!")
    else:
        print("It's a draw!")

if __name__ == "__main__":
    play_game()

