'''
Here is the environment of our PACMAN game, it was inspired by the Pacman environment of Freegames : 
https://grantjenks.com/docs/freegames/pacman.html
with modifications like adding a step function, changing the reward, adding a render function ...
'''
import copy
from ctypes import pointer
import numpy as np
from colorama import Fore, Back, Style

from random import choice
from turtle import *
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from sklearn import neighbors

from utils import floor, vector

class PacManEnv2(gym.Env):
    def __init__(self, size = "medium"):
        super().__init__()
        self.action_space = spaces.Discrete(5) # go towards L, R, U, D, STOP

        self.win = False #know if the pacman won

        self.total_reward = 0 #keep track of the total reward
        self.reward = 0 #update state reward

        self.size = size #What size of grid
        self.last_ghost = [] #keep track of the last position of the ghosts

        self.done = False # Know if the game is finished
            

        self.aim = vector(1, 0)

        if size == 'small':
            #set initial position of pacman and ghost
            self.pacman = vector(7,10)
            self.ghosts = [[vector(3,10), vector(1, 0)]]

            #set grid (keep 20 columns for render)
            self.tiles2D = np.array([
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ])

        elif size == 'medium':
            #set initial position of pacman and ghost
            self.pacman = vector(9,12)
            self.ghosts = [
                [vector(2, 1), vector(0, 1)],
                [vector(5, 10), vector(0, 1)],
            ]

            #set grid (keep 20 columns for render)
            self.tiles2D = np.array([
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                [0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                [0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                [0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                [0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ])
            
        self.shape = self.tiles2D.shape 
        self.tiles = self.tiles2D.flatten()


    def valid(self,point): 
        """Return True if point is valid in tiles. (Not going into the walls)"""
        x,y = point
        if self.tiles2D[x,y] == 0:
            return False
        return True


    def move(self):
        """Move pacman and all ghosts."""
        if self.valid(self.pacman + self.aim):
            self.pacman.move(self.aim)
        else:
            self.reward = -5
        
        if self.tiles2D[self.pacman.x,self.pacman.y] == 2:
            self.reward = - 1

        if self.tiles2D[self.pacman.x,self.pacman.y] == 1:
            self.tiles2D[self.pacman.x,self.pacman.y] = 2
            self.reward = 10

        for point, course in self.ghosts:
            if self.valid(point + course):
                point.move(course)
            else:
                options = [
                    vector(1, 0),
                    vector(-1, 0),
                    vector(0, 1),
                    vector(0, -1),
                ]
                plan = choice(options)
                course.x = plan.x
                course.y = plan.y

        self.total_reward += self.reward

        x_p, y_p = self.pacman.x , self.pacman.y

        for point, course in (self.ghosts + self.last_ghost):

            x_g, y_g = point.x , point.y
            if (x_g == x_p and y_g == y_p) or (x_g-course.x == x_p and y_g-course.y == y_p):
                self.reward = -100
                #print("JE SUIS MORT A CAUSE D'UN GHOST TAMER")
                self.done = True
                self.total_reward += self.reward
                return
            self.last_ghost = copy.deepcopy(self.ghosts)

        if 1 not in self.tiles2D:
          self.reward = 100
          #print("J'AI GAGNé OUAIS'")
          self.win = True
          self.done = True
          self.total_reward += self.reward
          return

    def change(self,x, y):
        """Change pacman aim if valid."""
        if self.valid(self.pacman + vector(x, y)):
            self.aim.x = x
            self.aim.y = y

    def get_state(self, DQN = False):
        obs_space = self.tiles2D.copy()

        #get the position of the ghost and the pacman in obs_space
        obs_space[self.pacman.x,self.pacman.y] = 3
        for point, _ in self.ghosts:
            obs_space[point.x,point.y] = 4
        
        if DQN :
            return obs_space.flatten()

        #Now we create the state that we want :
        new_obs = []
        new_obs += self.get_nearest_(obs_space, 1) #fruit is type 1
        new_obs += self.get_nearest_(obs_space, 4) #ghost is type 4
        new_obs += self.get_walls(obs_space)

        return new_obs

    def step(self, action, DQN = False):

        self.reward =  0
        if action == 0:
            self.change(-1,0)
        elif action == 1:
            self.change(1,0)
        elif action == 2:
            self.change(0,1)
        elif action == 3:
            self.change(0,-1)
        elif action == 4:
            self.change(0,0)

        self.move()
        
        state = self.get_state(DQN)
        return tuple(state),self.reward, self.done  

    def get_pos(self, obs_space, cat):
        elements = []
        for i in range(len(obs_space)):
            for j in range(len(obs_space[0])):
                if obs_space[i,j] == cat:
                    elements.append((i,j))
        
        return elements

    def get_nearest_(self, obs_space, cat):
        x, y = self.pacman.x, self.pacman.y
        min_dist = np.inf
        dx,dy = 0,0
        elements = self.get_pos(obs_space, cat)
        for x_f,y_f in elements:
            dist = np.sqrt((x-x_f)**2 + (y-y_f)**2)
            if dist < min_dist:
                min_dist = dist
                dx,dy = (x-x_f), (y-y_f)
        return dx, dy

    def get_walls(self, obs_space):
        x, y = self.pacman.x, self.pacman.y

        wall_left = obs_space[x-1][y] == 0
        wall_right = obs_space[x+1][y] == 0
        wall_up = obs_space[x][y-1] == 0
        wall_down = obs_space[x][y+1] == 0

        return int(wall_left), int(wall_right), int(wall_up), int(wall_down)
    
    #Render functions


    def render_init(self):
        self.path = Turtle(visible=False) 
        self.writer = Turtle(visible=False)



    
    def offset(self, point):
        """Return offset of point in tiles."""
        index = (point.x) * 20 + point.y
        x = (index % 20) * 20 - 200
        y = 180 - (index // 20) * 20
        return x,y
    
    def square(self,x, y):
        """Draw square using path at (x, y)."""
        self.path.up()
        self.path.goto(x, y)
        self.path.down()
        self.path.begin_fill()

        for count in range(4):
            self.path.forward(20)
            self.path.left(90)

        self.path.end_fill()

    
    def world(self):
        """Draw world using path."""
        bgcolor('black')
        self.path.color('blue')

        self.tiles = self.tiles2D.flatten()

        for index in range(len(self.tiles)):
            tile = self.tiles[index]

            if tile > 0:
                x = (index % 20) * 20 - 200
                y = 180 - (index // 20) * 20
                self.square(x, y)

                if tile == 1:
                    self.path.up()
                    self.path.goto(x + 10, y + 10)
                    self.path.dot(2, 'white')

    

    def rendertext(self):
        obs_space = self.tiles2D.copy()

        #get the position of the ghost and the pacman in obs_space
        obs_space[self.pacman.x,self.pacman.y] = 3
        for point, _ in self.ghosts:
            obs_space[point.x,point.y] = 4

        for line in obs_space:
            strout = ""
            for case in line:
                if case == 0:
                    strout += Fore.BLUE + str(case) + " "
                elif case == 1:
                    strout += Fore.GREEN + str(case) + " "
                elif case == 2:
                    strout += Fore.WHITE + str(case) + " "
                elif case == 3:
                    strout += Fore.BLACK + str(case) + " "
                else:
                    strout += Fore.RED + str(case) + " "
            print(strout)
        print('\n\n')

    def render(self):
        self.writer.undo()
        self.writer.write(self.total_reward)

        clear()

        #Make the fruits disapear where pacman was
        x,y = self.offset(self.pacman)
        self.square(x, y)

        #Add pacman
        up()
        goto(x + 10, y + 10)
        dot(20, 'yellow')

        #Add ghosts
        for point, course in self.ghosts :
            x,y = self.offset(point)
            up()
            goto(x + 10, y + 10)
            dot(20, 'red')

 
    def reset(self):
        self.__init__(self.size)
        return tuple(self.tiles2D.copy().flatten())