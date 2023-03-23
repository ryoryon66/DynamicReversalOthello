import tkinter as tk
from webbrowser import Opera
from PIL import Image, ImageTk
import numpy as np
from othelloRL.othelloenv import OthelloEnv
from othelloRL.agent import DDQN_Agent
import tkinter.messagebox
import colorsys

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from collections import deque
import threading
import time

from enum import Enum

weight_path = ""

TIME = 5.0

class Operator(Enum):
    CPU = 1
    HUMAN = 2

class OthelloGUI:
    def __init__(self, board_size=8 ):
        self.board_size = board_size
        self.env = OthelloEnv(board_size=board_size)
        self.env.player = Operator.CPU.value
        self.cpu_agent = DDQN_Agent(weight_path,self.env)
        self.gui_board = np.zeros((board_size, board_size), dtype=object)
        self.player_colors = {1: "#000000", 2: "#FFFFFF"}
        
        self.valid_moves_color = "#00FF00"
        self.highlight_color = "#FFFF00"
        self.init_gui()



    def init_gui(self):
        self.window = tk.Tk()
        self.window.title("DynamicReversalOthello")
        self.window.resizable(False, False)

        # Create the game board
        for i in range(self.board_size):
            for j in range(self.board_size):
                cell = tk.Canvas(self.window, width=64, height=64, highlightthickness=0)
                cell.grid(row=i, column=j)
                cell.bind("<Button-1>", lambda e, i=i, j=j: self.on_click(i, j))
                self.gui_board[i, j] = cell
                self.draw_cell(i, j)

        # Create the score label
        self.score_label = tk.Label(self.window, text="Human(WHITE): 0   CPU(BLACK): 0", font=("Arial", 16))
        self.score_label.grid(row=self.board_size, columnspan=self.board_size)
        
        
        self.time_label = tk.Label(self.window, text=f"Time left: {TIME}", font=("Arial", 16))
        self.time_label.grid(row=self.board_size + 1, columnspan=self.board_size)
        self.time_left = TIME
        self.timer_thread = threading.Thread(target=self.update_timer)
        self.timer_thread.daemon = True
        self.timer_thread.start()
        

        # Start the game loop
        self.update_gui()
        self.window.mainloop()
        
    def update_gui(self):
        
        # Check if it's CPU's turn
        if self.env.player == Operator.CPU.value:
            self.window.after(1000, self.cpu_move)

        # Update the board
        state = self.env.get_state()
        for i in range(self.board_size):
            for j in range(self.board_size):
                self.draw_cell(i, j)
                if state[i, j] != 0:
                    self.gui_board[i, j].create_oval(8, 8, 56, 56, fill=self.player_colors[state[i, j]])
                else:
                    self.remove_highlight(i, j)

        # Update the valid moves
        valid_moves = self.env.get_valid_actions()
        for i, j in valid_moves:
            self.draw_valid_move(i, j)

        # Update the score
        self.score_label.config(text=f"Human(white): {self.env.scores[Operator.HUMAN.value]}   CPU(black): {self.env.scores[Operator.CPU.value]}")
        
        
        if self.env.is_game_over():
            winner = self.env.get_winner()
            if winner == Operator.HUMAN.value:
                winner_text = "Human wins!"
            elif winner == Operator.CPU.value:
                winner_text = "CPU wins!"
            else:
                winner_text = "It's a draw!"
            tk.messagebox.showinfo("Game Over", winner_text)


    def on_click(self, i, j):
        if self.env.player == Operator.HUMAN.value:
            if self.env.is_valid_move(i, j):
                self.time_left = TIME
                self.env.step((i, j), player=Operator.HUMAN.value)
                self.update_gui()

    def draw_cell(self, i, j):
        if (i + j) % 2 == 0:
            color = "#008000"
        else:
            color = "#006400"
        self.gui_board[i, j].create_rectangle(0, 0, 64, 64, fill=color)

    def draw_valid_move(self, i, j):
        self.gui_board[i, j].create_oval(8, 8, 56, 56, fill=self.valid_moves_color)

    def draw_highlight(self, i, j):
        self.gui_board[i, j].create_oval(8, 8, 56, 56, outline=self.highlight_color, width=3)

    def remove_highlight(self, i, j):
            self.gui_board[i, j].delete("highlight")

    def cpu_move(self):
        
        if self.env.is_game_over():
            winner = self.env.get_winner()
            if winner == Operator.HUMAN.value:
                winner_text = "Human wins!"
            elif winner == Operator.CPU.value:
                winner_text = "CPU wins!"
            else:
                winner_text = "It's a draw!"
            tk.messagebox.showinfo("Game Over", winner_text)
            
        action = self.cpu_action(self.env)
        if action is not None:
            x, y = action
        if action is not None and self.env.is_valid_move(x, y):
            self.env.step(action, player=Operator.CPU.value)
            self.update_gui()
            
        if self.env.is_game_over():
            winner = self.env.get_winner()
            if winner == Operator.HUMAN.value:
                winner_text = "Human wins!"
            elif winner == Operator.CPU.value:
                winner_text = "CPU wins!"
            else:
                winner_text = "It's a draw!"
            tk.messagebox.showinfo("Game Over", winner_text)
        
        self.env.render()
            
    def cpu_action(self,env : OthelloEnv):
        
        assert (env.get_state() == self.cpu_agent.env.get_state()).all()
        
        state = env.get_state()
    
        valid_actions = env.get_valid_actions()
        
        if len(valid_actions) == 0:
            return None
        
        print ("valid actions DDQN : ",valid_actions)
        evaluation = self.cpu_agent.model(Variable(torch.from_numpy(env.get_state()).unsqueeze(0).float()).to(self.cpu_agent.device))
        valid_actions_encoded = [self.cpu_agent._encode_action(action) for action in valid_actions]
        evaluation = evaluation[0,valid_actions_encoded]
        print("evaluation : ",evaluation.tolist())
        action = self.cpu_agent.get_action(state)
        
        
        return action
    
    

    def update_timer(self):
        while True:
            # print (self.time_left)
            if self.env.player == Operator.HUMAN.value:
                self.time_left -= 0.01
                # 残り時間に応じてテキスト色を変更
                fraction = (self.time_left / (TIME)  ) ** 2
                hue =  0.33 * fraction # Hueを0.33から0に変化させる
                # print(hue,fraction)
                r, g, b = colorsys.hls_to_rgb(hue, 0.5, 1)
                color = f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
                self.time_label.config(fg=color)

                if self.time_left <= 0:
                    self.time_left = 0
                    self.env.player = Operator.CPU.value
                    tk.messagebox.showwarning("Time's up!", "Time's up! You lose this turn.")
                self.time_label.config(text=f"Time left: {round(self.time_left, 1)}")
            else:
                self.time_left = TIME
            time.sleep(0.01)
            
            if self.env.is_game_over():
                break

if __name__ == "__main__":
    OthelloGUI()
