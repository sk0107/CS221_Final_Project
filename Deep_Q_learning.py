import gym
import numpy as np
import tensorflow as tf
from ale_py import ALEInterface

STEPS = 1000
EPSILON = 0.1
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.99

class QNetwork(tf.keras.Model):
    def __init__(self, action_space_size):
        super(QNetwork, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(210, 160, 3))
        self.conv2 = tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(512, activation='relu')
        self.fc2 = tf.keras.layers.Dense(action_space_size)

    def call(self, state):
        x = self.conv1(state)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return self.fc2(x)

def epsilon_greedy_policy(Q, state):
    if np.random.rand() < EPSILON:
        return np.random.randint(Q.action_space_size)  # Explore
    else:
        return np.argmax(Q(state.reshape(1, *state.shape)))  # Exploit

def main():
    ale = ALEInterface()
    ale.loadROM('roms/SpaceInvaders.bin')
    env = gym.make('SpaceInvaders-v4', render_mode="human")

    action_space_size = env.action_space.n
    observation_space_size = (210, 160, 3)

    # Initialize DQN
    Q = QNetwork(action_space_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    for episode in range(STEPS):
        observation = env.reset()
        observation = observation.reshape(1, *observation_space_size)
        terminated = False

        while not terminated:
            env.render()

            # Choose action using epsilon-greedy policy
            action = epsilon_greedy_policy(Q, observation)

            # Take the chosen action
            next_observation, reward, terminated, _, _ = env.step(action)
            next_observation = next_observation.reshape(1, *observation_space_size)

            # Update Q-value using DQN
            with tf.GradientTape() as tape:
                Q_values = Q(observation)
                Q_action = Q_values[0, action]
                target = reward + DISCOUNT_FACTOR * np.max(Q(next_observation))
                loss = tf.keras.losses.mean_squared_error(target, Q_action)

            gradients = tape.gradient(loss, Q.trainable_variables)
            optimizer.apply_gradients(zip(gradients, Q.trainable_variables))

            observation = next_observation

            if terminated:
                print(f"Episode {episode + 1} completed with total reward: {np.sum(Q_values)}")

    env.close()

if __name__ == "__main__":
    main()
