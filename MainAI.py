import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import csv
from bayes_opt import BayesianOptimization
import random

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the PPO Policy network with dynamic architecture
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes):
        super(PolicyNet, self).__init__()
        layers = []
        input_size = state_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        layers.append(nn.Linear(input_size, action_dim))
        self.fc = nn.Sequential(*layers)
    
    def forward(self, x):
        logits = self.fc(x)
        return torch.softmax(logits, dim=-1)

# Define the PPO Value network with dynamic architecture
class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_sizes):
        super(ValueNet, self).__init__()
        layers = []
        input_size = state_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        layers.append(nn.Linear(input_size, 1))
        self.fc = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.fc(x)

# Define the DQN network with dynamic architecture
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes):
        super(DQN, self).__init__()
        layers = []
        input_size = state_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        layers.append(nn.Linear(input_size, action_dim))
        self.fc = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.fc(x)

# Define the Intrinsic Curiosity Module with dynamic architecture
class ICM(nn.Module):
    def __init__(self, state_dim, action_dim, feature_sizes, inverse_hidden_sizes, forward_hidden_sizes):
        super(ICM, self).__init__()
        # Feature Extraction
        feature_layers = []
        input_size = state_dim
        for size in feature_sizes:
            feature_layers.append(nn.Linear(input_size, size))
            feature_layers.append(nn.ReLU())
            input_size = size
        self.feature = nn.Sequential(*feature_layers)
        
        # Inverse Model
        inverse_layers = []
        input_inverse = 2 * input_size  # phi_state concatenated with phi_next_state
        for size in inverse_hidden_sizes:
            inverse_layers.append(nn.Linear(input_inverse, size))
            inverse_layers.append(nn.ReLU())
            input_inverse = size
        inverse_layers.append(nn.Linear(input_inverse, action_dim))
        self.inverse = nn.Sequential(*inverse_layers)
        
        # Forward Model
        forward_layers = []
        input_forward = input_size + action_dim  # phi_state concatenated with action_onehot
        for size in forward_hidden_sizes:
            forward_layers.append(nn.Linear(input_forward, size))
            forward_layers.append(nn.ReLU())
            input_forward = size
        forward_layers.append(nn.Linear(input_forward, input_size))
        self.forward_model = nn.Sequential(*forward_layers)
    
    def forward(self, state, next_state, action_onehot):
        phi_state = self.feature(state)
        phi_next_state = self.feature(next_state)
        
        # Inverse model
        inverse_input = torch.cat([phi_state, phi_next_state], dim=1)
        pred_action = self.inverse(inverse_input)
        
        # Forward model
        forward_input = torch.cat([phi_state, action_onehot], dim=1)
        pred_phi_next_state = self.forward_model(forward_input)
        
        return pred_action, pred_phi_next_state, phi_next_state

# Replay buffer for DQN
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
    
    def push(self, transition):
        self.buffer.append(transition)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        return batch
    
    def __len__(self):
        return len(self.buffer)

def train_and_evaluate(
    learning_rate,             # PPO and DQN learning rate
    icm_lr,                    # ICM learning rate
    hidden_size_policy,        # Size of hidden layers for PolicyNet
    hidden_size_value,         # Size of hidden layers for ValueNet
    hidden_size_dqn,           # Size of hidden layers for DQN
    feature_size_icm,          # ICM feature size
    inverse_hidden_size_icm,   # ICM inverse model hidden sizes
    forward_hidden_size_icm,   # ICM forward model hidden sizes
    ppo_epochs,                # Number of PPO epochs
    ppo_clip,                  # PPO clipping parameter
    icm_beta                   # Weight for intrinsic reward
):
    # Set random seeds for reproducibility
    random_seed = 42
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    env = gym.make('CartPole-v1')
    env.action_space.seed(random_seed)
    env.observation_space.seed(random_seed)

    # Hyperparameters
    num_episodes = 1000  # Reduced for quicker evaluation
    gamma = 0.99
    lam = 0.95  # GAE lambda for PPO
    dqn_batch_size = 64
    replay_size = 10000

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Define hidden_sizes as lists based on hidden_size input
    hidden_sizes_policy = [int(hidden_size_policy)]  # Single hidden layer
    hidden_sizes_value = [int(hidden_size_value)]    # Single hidden layer
    hidden_sizes_dqn = [int(hidden_size_dqn)]        # Single hidden layer
    feature_sizes_icm = [int(feature_size_icm)]
    inverse_hidden_sizes_icm = [int(inverse_hidden_size_icm)]  # Single hidden layer
    forward_hidden_sizes_icm = [int(forward_hidden_size_icm)]  # Single hidden layer

    # Initialize Networks
    policy_net = PolicyNet(state_dim, action_dim, hidden_sizes_policy).to(device)
    value_net = ValueNet(state_dim, hidden_sizes_value).to(device)
    dqn_net = DQN(state_dim, action_dim, hidden_sizes_dqn).to(device)
    target_dqn_net = DQN(state_dim, action_dim, hidden_sizes_dqn).to(device)
    target_dqn_net.load_state_dict(dqn_net.state_dict())
    icm = ICM(state_dim, action_dim, feature_sizes_icm, inverse_hidden_sizes_icm, forward_hidden_sizes_icm).to(device)

    # Initialize Optimizers
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    value_optimizer = optim.Adam(value_net.parameters(), lr=learning_rate)
    dqn_optimizer = optim.Adam(dqn_net.parameters(), lr=learning_rate)
    icm_optimizer = optim.Adam(icm.parameters(), lr=icm_lr)

    # Initialize Replay Buffer
    replay_buffer = ReplayBuffer(replay_size)

    # CSV Logging setup (Optional: Can be commented out if not needed)
    log_filename = f'training_log_lr_{learning_rate:.5f}_icm_lr_{icm_lr:.5f}_hidden_p_{hidden_size_policy}_v_{hidden_size_value}_dqn_{hidden_size_dqn}_feature_{feature_size_icm}.csv'
    with open(log_filename, mode='w', newline='') as csv_file:
        fieldnames = ['Episode', 'Total Reward', 'Actor Loss', 'Critic Loss', 'DQN Loss', 'ICM Loss']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        rewards = []  # Initialize a list to store rewards

        for episode in range(1, num_episodes + 1):
            state, _ = env.reset()
            done = False
            total_reward = 0
            transitions = []
            episode_steps = 0

            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action_probs = policy_net(state_tensor)
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample().item()

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # Intrinsic Curiosity Module
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
                action_onehot = torch.zeros((1, action_dim)).to(device)
                action_onehot[0][action] = 1

                pred_action_logits, pred_phi_next_state, phi_next_state = icm(
                    state_tensor, next_state_tensor, action_onehot
                )
                inverse_loss = nn.CrossEntropyLoss()(pred_action_logits, torch.tensor([action]).to(device))
                forward_loss = nn.MSELoss()(pred_phi_next_state, phi_next_state.detach())
                icm_loss = (1 - icm_beta) * inverse_loss + icm_beta * forward_loss

                icm_optimizer.zero_grad()
                icm_loss.backward()
                icm_optimizer.step()

                intrinsic_reward = icm_beta * forward_loss.item()
                total_reward += np.clip(reward + intrinsic_reward, -1e3, 1e3)

                # Store transition for PPO
                value = value_net(state_tensor).item()
                transitions.append((state, action, action_probs.data.squeeze().cpu().numpy(), value, reward + intrinsic_reward, done))
                
                # Store transition for DQN
                replay_buffer.push((state, action, reward + intrinsic_reward, next_state, done))

                state = next_state
                episode_steps += 1

                # DQN update
                if len(replay_buffer) >= dqn_batch_size:
                    batch = replay_buffer.sample(dqn_batch_size)
                    batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = zip(*batch)
                    batch_states = torch.FloatTensor(batch_states).to(device)
                    batch_actions = torch.LongTensor(batch_actions).unsqueeze(1).to(device)
                    batch_rewards = torch.FloatTensor(batch_rewards).unsqueeze(1).to(device)
                    batch_next_states = torch.FloatTensor(batch_next_states).to(device)
                    batch_dones = torch.FloatTensor(batch_dones).unsqueeze(1).to(device)

                    q_values = dqn_net(batch_states).gather(1, batch_actions)
                    with torch.no_grad():
                        next_q_values = target_dqn_net(batch_next_states).max(1, keepdim=True)[0]
                        target_q_values = batch_rewards + gamma * next_q_values * (1 - batch_dones)
                    dqn_loss = nn.MSELoss()(q_values, target_q_values)

                    dqn_optimizer.zero_grad()
                    dqn_loss.backward()
                    dqn_optimizer.step()

                    # Update target network
                    for param, target_param in zip(dqn_net.parameters(), target_dqn_net.parameters()):
                        target_param.data.copy_(0.995 * target_param.data + 0.005 * param.data)
                else:
                    dqn_loss = torch.tensor(0.0)

            # Compute returns and advantages for PPO
            returns = []
            advantages = []
            G = 0
            adv = 0
            for t in reversed(range(len(transitions))):
                state_t, action_t, action_prob_t, value_t, reward_t, done_t = transitions[t]
                G = reward_t + gamma * G * (1 - done_t)
                if t < len(transitions) - 1:
                    next_value = value_net(torch.FloatTensor(transitions[t + 1][0]).unsqueeze(0).to(device)).item()
                else:
                    next_value = 0
                delta = reward_t + gamma * next_value * (1 - done_t) - value_t
                adv = delta + gamma * lam * adv * (1 - done_t)
                returns.insert(0, G)
                advantages.insert(0, adv)
            returns = torch.FloatTensor(returns).to(device)
            advantages = torch.FloatTensor(advantages).to(device)
            states = torch.FloatTensor([t[0] for t in transitions]).to(device)
            actions = torch.LongTensor([t[1] for t in transitions]).to(device)
            old_action_probs = torch.FloatTensor([t[2] for t in transitions]).to(device)

            # PPO Policy update
            for _ in range(int(ppo_epochs)):
                action_probs = policy_net(states)
                dist = torch.distributions.Categorical(action_probs)
                new_log_probs = dist.log_prob(actions)
                old_log_probs = torch.log(old_action_probs.gather(1, actions.unsqueeze(1)).squeeze(1) + 1e-10)

                ratio = torch.exp(new_log_probs - old_log_probs.detach())
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - ppo_clip, 1 + ppo_clip) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                policy_optimizer.zero_grad()
                actor_loss.backward()
                policy_optimizer.step()

                # PPO Value update
                values = value_net(states).squeeze(1)
                critic_loss = nn.MSELoss()(values, returns)

                value_optimizer.zero_grad()
                critic_loss.backward()
                value_optimizer.step()

            # Logging
            writer.writerow({
                'Episode': episode,
                'Total Reward': total_reward,
                'Actor Loss': actor_loss.item(),
                'Critic Loss': critic_loss.item(),
                'DQN Loss': dqn_loss.item(),
                'ICM Loss': icm_loss.item()
            })
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Actor Loss: {actor_loss.item():.4f}, "
                  f"Critic Loss: {critic_loss.item():.4f}, DQN Loss: {dqn_loss.item():.4f}, ICM Loss: {icm_loss.item():.4f}")
            rewards.append(total_reward)

    # Compute average reward over the last 100 episodes
    average_reward = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
    print(f"Average Reward over last 100 episodes: {average_reward}")
    if not np.isfinite(average_reward):
        average_reward = -1e10
    return average_reward

if __name__ == "__main__":
    # Define the hyperparameter bounds for Bayesian Optimization, including architecture parameters
    pbounds = {
        'learning_rate': (1e-4, 1e-2),            # PPO and DQN learning rate
        'icm_lr': (1e-4, 1e-2),                   # ICM learning rate
        'hidden_size_policy': (64, 256),          # Size of hidden layers for PolicyNet
        'hidden_size_value': (64, 256),           # Size of hidden layers for ValueNet
        'hidden_size_dqn': (64, 256),             # Size of hidden layers for DQN
        'feature_size_icm': (64, 256),            # ICM feature size
        'inverse_hidden_size_icm': (128, 512),    # ICM inverse model hidden size
        'forward_hidden_size_icm': (128, 512),    # ICM forward model hidden size
        'ppo_epochs': (1, 10),                    # Number of PPO epochs
        'ppo_clip': (0.1, 0.3),                   # PPO clipping parameter
        'icm_beta': (0.1, 0.5)                    # Weight for intrinsic reward
    }

    # Define the objective function for Bayesian Optimization
    def objective(learning_rate, icm_lr, hidden_size_policy, hidden_size_value,
                  hidden_size_dqn, feature_size_icm, inverse_hidden_size_icm,
                  forward_hidden_size_icm, ppo_epochs, ppo_clip, icm_beta):
        # Convert float hyperparameters to integers where necessary
        hidden_size_policy = int(hidden_size_policy)
        hidden_size_value = int(hidden_size_value)
        hidden_size_dqn = int(hidden_size_dqn)
        feature_size_icm = int(feature_size_icm)
        inverse_hidden_size_icm = int(inverse_hidden_size_icm)
        forward_hidden_size_icm = int(forward_hidden_size_icm)
        ppo_epochs = int(ppo_epochs)
        
        try:
            average_reward = train_and_evaluate(
                learning_rate, icm_lr, hidden_size_policy, hidden_size_value,
                hidden_size_dqn, feature_size_icm, inverse_hidden_size_icm,
                forward_hidden_size_icm, ppo_epochs, ppo_clip, icm_beta
            )
            return average_reward  # Bayesian Optimization will try to maximize this
        except Exception as e:
            print(f"Exception during training: {e}")
            return -1e10  # Penalize the optimizer if training fails

    # Initialize Bayesian Optimization
    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        random_state=42,
        verbose=2
    )

    # Perform Bayesian Optimization
    optimizer.maximize(
        init_points=5,  # Number of initial random exploration steps
        n_iter=15      # Number of optimization steps
    )

    # Print the best result
    print("Best Hyperparameters Found:")
    print(optimizer.max)

    # Extract the best hyperparameters
    best_params = optimizer.max['params']
    best_params['hidden_size_policy'] = int(best_params['hidden_size_policy'])
    best_params['hidden_size_value'] = int(best_params['hidden_size_value'])
    best_params['hidden_size_dqn'] = int(best_params['hidden_size_dqn'])
    best_params['feature_size_icm'] = int(best_params['feature_size_icm'])
    best_params['inverse_hidden_size_icm'] = int(best_params['inverse_hidden_size_icm'])
    best_params['forward_hidden_size_icm'] = int(best_params['forward_hidden_size_icm'])
    best_params['ppo_epochs'] = int(best_params['ppo_epochs'])

    # Run the training loop with the best hyperparameters
    train_and_evaluate(
        best_params['learning_rate'], best_params['icm_lr'], best_params['hidden_size_policy'],
        best_params['hidden_size_value'], best_params['hidden_size_dqn'], best_params['feature_size_icm'],
        best_params['inverse_hidden_size_icm'], best_params['forward_hidden_size_icm'],
        best_params['ppo_epochs'], best_params['ppo_clip'], best_params['icm_beta']
    )
