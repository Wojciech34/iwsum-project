import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import flappy_bird_gymnasium
import gymnasium as gym
import logging

logging.basicConfig(
    filename="basic_dqn.txt",
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
learning_rate = 0.00001
gamma = 0.99
batch_size = 64
memory_size = 10000
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995
target_update_freq = 10
save_model_freq = 50
num_episodes = 12000

# Replay memory
memory = deque(maxlen=memory_size)

# Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_shape, n_actions):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_shape, 256)
        # self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, n_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # No softmax for Q-values
        return x

# Instantiate the networks
policy_net = QNetwork(state_shape, n_actions).to(device)
target_net = QNetwork(state_shape, n_actions).to(device)
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

# Add experience to replay memory
def store_experience(memory, state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

# Sample a batch of experiences
def sample_batch(memory, batch_size):
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    states = torch.tensor(states, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.int64).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).to(device)
    return states, actions, rewards, next_states, dones

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
        store_experience(memory, state, action, reward, next_state, done)

        # Update state and accumulate reward
        state = next_state
        total_reward += reward

        # Train the policy network
        if len(memory) >= batch_size:
            states, actions, rewards, next_states, dones = sample_batch(memory, batch_size)
            q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                max_next_q_values = target_net(next_states).max(dim=1)[0]
                targets = rewards + gamma * max_next_q_values * (1 - dones)
            loss = nn.MSELoss()(q_values, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Decay epsilon
    epsilon = max(epsilon_end, epsilon * epsilon_decay)

    # Update target network periodically
    if episode % target_update_freq == 0:
        target_net.load_state_dict(policy_net.state_dict())

    logging.info(f'{episode} {total_reward}')

    if episode % save_model_freq == 0:
        print(f"Episode {episode}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")
        torch.save(target_net, "basic_dqn.pth")

env.close()
