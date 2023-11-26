import gym
import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam
from collections import deque

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=self.state_size))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < 100:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward if done else reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def preprocess_state(state):
    # Implement any preprocessing here
    # For simplicity, this example does not include preprocessing
    return state

def main():
    env = gym.make('SpaceInvaders-v4', render_mode="human")
    state_size = (210, 160, 3)  # Update with correct dimensions if different
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    batch_size = 16

    episodes = 1000  # Set total number of episodes
    for e in range(episodes):
        state = preprocess_state(env.reset())
        state = state[0] 
        state = np.reshape(state, [1, state_size[0], state_size[1], state_size[2]])
        terminated = False
        train_interval = 10  # Train after every 10 steps
        step_count = 0

        while not terminated:
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = preprocess_state(next_state)
            next_state = np.reshape(next_state, [1, state_size[0], state_size[1], state_size[2]])
            agent.remember(state, action, reward, next_state, terminated)
            state = next_state

            step_count += 1
            if step_count % train_interval == 0 and len(agent.memory) > batch_size:
                agent.replay(batch_size)

            if terminated:
                observation = env.reset()

                # print("Episode: {}/{}, Score: {}".format(e, episodes, info['score']))
                break

            if step_count % train_interval == 0 and len(agent.memory) > batch_size:
                agent.replay(batch_size)

    env.close()

if __name__ == "__main__":
    main()
