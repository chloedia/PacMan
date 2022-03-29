from pacman_env_v2 import *
from tqdm import tqdm
import random
from IPython.display import clear_output
from collections import namedtuple, deque
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

########################## replay memory #########################
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([],maxlen = capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

##########################  CNN  #########################
class DQN(nn.Module):
    def __init__(self, outputs=4):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=4)
        self.flat = nn.Linear(1664, 256)
        self.output = nn.Linear(256,outputs)

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        #x = torch.flatten(x)
        x = F.relu(self.flat(x.view(x.size(0), -1)))
        return self.output(x)
    
########################## Agent #########################
class DQNAgent():
    def agent_init(self, agent_init_info, env):
        """Setup for the agent called when the experiment first starts.
        
        Args:
        agent_init_info (dict), the parameters used to initialize the agent. The dictionary contains:
        {
            num_states (int): The number of states,
            num_actions (int): The number of actions,
            epsilon (float): The epsilon parameter for exploration,
            step_size (float): The step-size,
            discount (float): The discount factor,
        }
        
        """
        # Store the parameters provided in agent_init_info.
        self.num_actions = agent_init_info["num_actions"]
        self.epsilon = agent_init_info["epsilon"]
        self.step_size = agent_init_info["step_size"]
        self.discount = agent_init_info["discount"]
        self.rand_generator = np.random.RandomState(agent_init_info["seed"])

        # Memory : storing the last 100,000 experiences
        self.memory = ReplayMemory(100000)
        self.env = env
        self.state = self.env.reset()
        self.action = [0,1,2,3,4]
        self.reward_history = []

        # Statistics
        self.episode_number = 0 
        self.episode_reward = 0
        self.counter = 0
        self.win_counter = 0

        self.epsilon_decay = 0.9
        self.epsilon_min = 0.01

        # DQN
        self.batch_size = 32
        # init model
        self.model = DQN().to(device)
        self.target_model = DQN().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()
            
    def agent_start(self):
        """The first method called when the episode starts, called after
        the environment starts.
        Args:
            state (int): the state from the
                environment's evn_start function.
        Returns:
            action (int): the first action the agent takes.
        """
        
        # Choose action using epsilon greedy.
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions) # random action selection
        else:
            outputs= torch.tensor(self.state).float().to(device)
            outputs = torch.reshape(outputs,(1,1,11,20))
            outputs = self.model(outputs)
            action = self.action[np.random.choice(torch.where(outputs == outputs.max()))]

        return action
     
    def set_epsilon(self, value):
        self.epsilon = value

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Get State
        state_batch = torch.tensor(batch.state).float().to(device)
        state_batch = torch.reshape(state_batch,(self.batch_size,1,11,20))

        # Get Action
        action_batch = torch.tensor(batch.action)

        # Get Q(s, a)
        state_action_values = self.model(state_batch).gather(1, action_batch.unsqueeze(1).to(device))

        # Get Next State
        next_state_batch = torch.tensor(batch.next_state).float().to(device)
        next_state_batch = torch.reshape(next_state_batch,(self.batch_size,1,11,20))
    
        # Get V(s')
        next_state_values = self.target_model(next_state_batch)

        # Compute the expected Q values                        
        next_state_values = next_state_values.detach().max(1)[0]
        next_state_values = next_state_values.unsqueeze(1)
                
        # Get Reward
        reward_batch = torch.tensor(batch.reward)#.float()
    
        # Loss
        expected_state_action_values = (next_state_values * 0.9) + reward_batch.unsqueeze(1)
        loss = self.criterion(state_action_values, expected_state_action_values.to(device))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

########################## Run Episode #########################

def run_episode(agent):
    agent.state = agent.env.reset()

    ## decrease epsilon :
    if agent.epsilon > agent.epsilon_min:
        epsilon_value = agent.epsilon * agent.epsilon_decay
        agent.set_epsilon(epsilon_value)

    while True:
        action = agent.agent_start()
        next_state, reward, done = agent.env.step(action, DQN = True) 
        
        agent.episode_reward += reward
        agent.memory.push(agent.state, action, next_state, reward)
        agent.train()
        agent.state = next_state
        if done:
            break

    agent.episode_number += 1
    agent.reward_history.append(agent.episode_reward)
    print("Episode no = ", str(agent.episode_number))
    print("Episode Reward: ", agent.episode_reward)
    print("Average Reward: ", np.mean(agent.reward_history))
    print()
    agent.episode_reward = 0
    agent.counter += 1

    return reward

##########################  main  #########################
 
if __name__ == '__main__':
    #setup(420, 420, 370, 0)
    #hideturtle()
    #tracer(False)

    ##### initiate environment
    env = PacManEnv2(size = "medium")
    env.reset()
    #env.writer.color('black')

    ###### main parameters
    number_of_episodes = 1000
    epsilon_value = 0.1
    epsilon_decay = 0.9 # every 5000 episodes
    agent_info = {"num_actions": 4 , "epsilon": epsilon_value, "step_size": 0.1, "discount": 1.0, "seed": 0}

    ##### initialize both agents
    agents_dict = {'DQN': DQNAgent()}

    for agent_name in agents_dict.keys():
        agents_dict[agent_name].agent_init(agent_info, env)

    #### play episode
    rewards = {}
    for name, agent in agents_dict.items():
        print(f"Start training {name}")
        rewards_dict = {}
        epsilon_value = 0.1
        agent.set_epsilon(epsilon_value)
        for episode in tqdm(range(number_of_episodes), position = 0):
            new_reward = run_episode(agent)
            rewards_dict[episode] = new_reward # if episodes terminates, save the obtained reward
                
        rewards[name] = rewards_dict

    
    #### play episode
    env = PacManEnv2(size = 'medium')
    number_of_episodes = 100
    rewards = {}
    for name, agent in agents_dict.items():
        print(f"Start playing {name}")
        rewards_dict = {}
        agent.set_epsilon(epsilon_value)
        for episode in tqdm(range(number_of_episodes), position = 0):
            new_reward = run_episode(agent)
            rewards_dict[episode] = new_reward # if episodes terminates, save the obtained reward
                
        rewards[name] = rewards_dict