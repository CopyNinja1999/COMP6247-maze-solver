import os
import torch
import numpy as np
from canvas import Canvas #framework from 
from read_maze import get_local_maze_information,load_maze
import pandas as pd
class Environment:
    """
    Provides all functions to interact with the environment
    """
    def __init__(self, goal_x=199, goal_y=199, start_x=1, start_y=1, fire=True,plot=True,canvas=None):
        self.timestep = 0
        self.maze_size = 201
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.x = start_x
        self.y = start_y
        self.fire = fire
        self.plot=plot
        self.action_space=["stay","left","right","up","down"]
        self.canvas=canvas
        self.actorpath=[(1,1)]
        self.around=get_local_maze_information(self.y, self.x)
        if not self.fire:
            self.around[:, :, 1] = 0
    def update_maze(self):
        """Update the plot and trace"""
        if self.plot:
            self.canvas.step(self.around, (self.x,self.y), self.actorpath)

    def get_vaild_actions(self):
        """
        Return all valid actions from current state. 
        Invalid actions include: Walk out of the maze, walk into the wall, walk into a fire.
        """
        # Stay
        valid_actions = ["stay"]
        # Left
        if self.around[1][0][0] == 1 and self.around[1][0][1] == 0 and self.x - 1 >= 0 and self.x - 1 < self.maze_size and self.y >= 0 and self.y < self.maze_size:
            valid_actions.append("left")
        # Right
        if self.around[1][2][0] == 1 and self.around[1][2][1] == 0 and self.x + 1 >= 0 and self.x + 1 < self.maze_size and self.y >= 0 and self.y < self.maze_size:
            valid_actions.append("right")
        # Up
        if self.around[0][1][0] == 1 and self.around[0][1][1] == 0 and self.x >= 0 and self.x < self.maze_size and self.y - 1 >= 0 and self.y - 1 < self.maze_size:
            valid_actions.append("up")
        # Down
        if self.around[2][1][0] == 1 and self.around[2][1][1] == 0 and self.x >= 0 and self.x < self.maze_size and self.y + 1 >= 0 and self.y + 1 < self.maze_size:
            valid_actions.append("down")
        return valid_actions

    def get_state(self):
        if len(self.get_vaild_actions())==2:
            return 0
            #blocked
        elif len(self.get_vaild_actions())==3:
            return 1
            #mormal 
        elif len(self.get_vaild_actions())>3:
            return 2
            #crossroad
    def get_next_position(self, action):
        """
        Return next position if action is taken. 
        Note that this is not really taking an action, the environment would not change.
        Also note that it does not care whether the action is valid or not.
        """
        if action == "stay":  # Stay
            x = self.x
            y = self.y
        elif action == "left":  # Left
            x = self.x - 1
            y = self.y
        elif action == "right":  # Right
            x = self.x + 1
            y = self.y
        elif action == "up":  # Up
            x = self.x
            y = self.y - 1
        elif action == "down":  # Down
            x = self.x
            y = self.y + 1
        else:
            raise ValueError(f"Unknown Action: {action}")
        
        return x, y

    def make_action(self, action):
        """
        Take 1 of the following actions: stay, left, right, up, down. 
        Increment timestep.
        Update environment states (x, y, around, maze).

        Return a reward: 0 if game ends, -1 otherwise.
        """
        reward = -1.0

        if action not in self.get_vaild_actions():  # If action is ilvalid, stay at current position and discount reward by 1
            action = "stay"
            reward -= 1

        self.x, self.y = self.get_next_position(action)
        #Tracking trace
        if self.plot:
            if len(self.actorpath)>10:
                self.actorpath.pop(0)
            self.actorpath.append((self.x,self.y))
            

        # Update agent states
        self.timestep += 1
        self.around = torch.tensor(get_local_maze_information(self.y, self.x))
        if not self.fire:
            self.around[:, :, 1] = 0
        self.update_maze()

        if self.isTerminated():
            return 0.0

        return reward

    def isTerminated(self):
        """Return True if agent reaches the goal point"""
        if self.x == self.goal_x and self.y == self.goal_y:  # 201 - 1 - wall
            return True
        return False

    def restart(self, x=1, y=1):
        """
        Move the agent to the starting position.
        Note that some fire might remain in the maze because we cannot call load_maze() again, 
        but they should be far away from the starting point so it does not really matter.
        """
        self.timestep = 0
        self.x = x
        self.y = y

        # Make initial state
        self.around = torch.tensor(get_local_maze_information(self.y, self.x))
        if not self.fire:
            self.around[:, :, 1] = 0
        self.update_maze()
"""
a base class for Q-learning, the table stores cumulative rewards for every action
"""
class RL(object):
    def __init__(self, action_space, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = action_space  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy

        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.rand() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, *args):
        pass

"""
backward eligibility traces
if an unseen state is observed, check_state_exist will add it to Q-table
"""

class Sarsa(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, trace_decay=0.9):
        super(Sarsa, self).__init__(actions, learning_rate, reward_decay, e_greedy)

        # backward view, eligibility trace.
        self.lambda_ = trace_decay
        self.eligibility_trace = self.q_table.copy()

    def check_state_exist(self, state):
        if str(state) not in self.q_table.index:
            # append new state to q table
            to_be_append = pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            self.q_table = self.q_table.append(to_be_append)

            # also update eligibility trace
            self.eligibility_trace = self.eligibility_trace.append(to_be_append)

    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        print(self.q_table)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, a_]  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        error = q_target - q_predict

        self.eligibility_trace.loc[s, :] *= 0
        self.eligibility_trace.loc[s, a] = 1

        # Q update
        self.q_table += self.lr * error * self.eligibility_trace

        # decay eligibility trace after update
        self.eligibility_trace *= self.gamma*self.lambda_
if __name__ == '__main__':
    #the load maze function is modified to return a matrix for building graph     
    maze=load_maze()
    # print(maze)
    canv = Canvas(maze)
    env=Environment(canvas=canv)
    RL = Sarsa(actions=env.action_space)
    # update()
    for episode in range(2):
        # initial observation
        #set up the canvas
        canv.set_visible(env.around, (env.x,env.y), [])
        env.restart()
        state=env.get_state()
        print(state)
        # RL choose action based on observation
        action = "right"
        valid_state=env.get_vaild_actions()
        RL.eligibility_trace *= 0
        while True:
            # fresh env
            env.update_maze()
            reward= env.make_action(action)
            state_=env.get_state()
            # RL choose action based on next observation
            action_ = RL.choose_action(str(state_))
            print("Time %d, location=(%d,%d), action=%s, state=%s, reward=%.2f"%(env.timestep,env.x,env.y,action,state,reward))
            RL.learn(str(state), action, reward, str(state_), action_)

            # swap observation and action
            state = state_
            action = action_

            # break while loop when end of this episode
            if env.isTerminated():
                break