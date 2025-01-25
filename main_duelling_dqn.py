import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import flappy_bird_gymnasium
import gymnasium as gym
import logging

logging.basicConfig(
    filename="duelling_dqn.txt",
    level=logging.INFO,
    format='%(message)s'
)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create the FlappyBird environment
env = gym.make("FlappyBird-v0", use_lidar=False)

# Hyperparameters
state_shape = env.observation_space.shape[0]
n_actions = env.action_space.n
learning_rate = 0.0001
gamma = 0.99
batch_size = 64
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995
target_update_freq = 10
save_model_freq = 50
num_episodes = 12000

# Q-Network (Dueling DQN)
class DuelingQNetwork(nn.Module):
    def __init__(self, state_shape, n_actions):
        super(DuelingQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_shape, 256)
        self.fc2 = nn.Linear(256, 256)
        
        # Value stream
        self.value_stream = nn.Linear(256, 1)
        
        # Advantage stream
        self.advantage_stream = nn.Linear(256, n_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        # Compute value and advantage
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Combine value and advantage to get Q-values
        q_values = value + (advantage - advantage.max()) # mean
        return q_values

# Instantiate the networks
policy_net = DuelingQNetwork(state_shape, n_actions).to(device)
target_net = DuelingQNetwork(state_shape, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# Optimizer
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

# Epsilon-greedy action selection
def select_action(state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, n_actions - 1)  # Explore
    else:
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = policy_net(state)
        return q_values.argmax(dim=1).item()  # Exploit

# Training Loop
epsilon = epsilon_start
for episode in range(1, num_episodes+1):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        # Select an action using epsilon-greedy policy
        action = select_action(state, epsilon)
        next_state, reward, done, truncated, _ = env.step(action)

        # Update state and accumulate reward
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Get the current Q-values and the target Q-values
        q_values = policy_net(state_tensor).squeeze(0)
        with torch.no_grad():
            next_q_values = target_net(next_state_tensor).max(1)[0].squeeze(0)
        
        target = reward + gamma * next_q_values * (1 - done)
        
        # Compute the loss
        loss = nn.MSELoss()(q_values[action], target)

        # Optimize the policy network
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update state
        state = next_state
        total_reward += reward

    # Decay epsilon
    epsilon = max(epsilon_end, epsilon * epsilon_decay)

    # Update target network periodically
    if episode % target_update_freq == 0:
        target_net.load_state_dict(policy_net.state_dict())
    
    logging.info(f'{episode} {total_reward}')

    if episode % save_model_freq == 0:
        print(f"Episode {episode}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")
        torch.save(target_net, "duelling_dqn.pth")

env.close()
