import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, action_size)
        )

    def forward(self, x):
        return self.network(x)


class Agent:
    rewards = []

    def __init__(self, state_size, is_eval=False, model_name=""):
        self.state_size = state_size  # normalized previous days
        self.action_size = 3  # sit, buy, sell
        self.memory = deque(maxlen=1000)
        self.inventory = []
        self.model_name = model_name
        self.is_eval = is_eval

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.delta = 0.97

        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if is_eval:
            self.model = torch.load(f"models/{model_name}")
        else:
            self.model = DQN(state_size, self.action_size).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            self.criterion = nn.MSELoss()

    def act(self, state):
        if not self.is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            options = self.model(state)
            return options.argmax().item()

    def stockRewards(self, rewardto):
        self.rewards.append(rewardto)

    def expReplay(self, batch_size):
        mini_batch = list(self.memory)[-batch_size:]

        for state, action, reward, next_state, done in mini_batch:
            target = reward

            if not done:
                state = torch.FloatTensor(state).to(self.device)
                next_state = torch.FloatTensor(next_state).to(self.device)

                with torch.no_grad():
                    target = reward + self.gamma * self.model(next_state).max(1)[0].item()

            state = torch.FloatTensor(state).to(self.device)
            target_f = self.model(state)
            target_f[0][action] = target

            self.optimizer.zero_grad()
            loss = self.criterion(self.model(state), target_f)
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def getRewards(self):
        rewards = []
        for state, action, reward, next_state, done in self.memory:
            if reward > 0:
                rewards.append(reward)
        return rewards

    def getAgentsrewards(self):
        return self.rewards


# Helper functions remain mostly the same
def formatPrice(n):
    return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))


def getStockDataVec(key):
    vec = []
    lines = open("data/" + key + ".csv", "r").read().splitlines()
    for line in lines[1:]:
        vec.append(float(line.split(",")[4]))
    return vec


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def getState(data, t, n):
    d = t - n + 1
    block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1]
    res = []
    for i in range(n - 1):
        res.append(sigmoid(block[i + 1] - block[i]))
    return np.array([res])


# Training loop
if __name__ == "__main__":
    stock_name, window_size, episode_count = 'GOLD', 3, 10
    agent = Agent(window_size)
    data = getStockDataVec(stock_name)
    l = len(data) - 1
    batch_size = 32

    total_profitl = []
    buy_info = []
    sell_info = []
    data_Store = []

    for e in range(episode_count + 1):
        print(f"Episode {e}/{episode_count}")
        state = getState(data, 0, window_size + 1)
        total_profit = 0
        agent.inventory = []

        for t in range(l):
            action = agent.act(state)
            next_state = getState(data, t + 1, window_size + 1)
            reward = 0

            if action == 1:  # buy
                agent.inventory.append(data[t])
                print(f"Buy: {formatPrice(data[t])}")
                buy_info.append(data[t])
                data_Store.append(f"{data[t]}, Buy")

            elif action == 2 and len(agent.inventory) > 0:  # sell
                bought_price = agent.inventory.pop(0)
                reward = max(data[t] - bought_price, 0)
                total_profit += data[t] - bought_price
                print(f"Sell: {formatPrice(data[t])} | Profit: {formatPrice(data[t] - bought_price)}")

                total_profitl.append(data[t] - bought_price)
                step_price = data[t] - bought_price
                sell_info.append(f"{data[t]},{step_price},{reward}")
                data_Store.append(f"{data[t]}, Sell")

            done = t == l - 1
            agent.memory.append((state, action, reward, next_state, done))
            state = next_state

            if done:
                print("--------------------------------")
                print(f"Total Profit: {formatPrice(total_profit)}")
                print("--------------------------------")

            if len(agent.memory) > batch_size:
                agent.expReplay(batch_size)

        if e % 10 == 0:
            torch.save(agent.model, f"models/model_ep{e}")