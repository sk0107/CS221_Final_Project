import gym
import numpy as np
import pickle
import random

def state_to_key(state):
    # Convert state to a immutable type (tuple of tuples) to be used as dictionary keys
    return pickle.dumps(state, protocol=0)

def initialize_q_table():
    return {}

def choose_action(state, q_table, action_space, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.choice(range(action_space))
    else:
        state_key = state_to_key(state)
        return np.argmax(q_table.get(state_key, np.zeros(action_space)))

def update_q_table(q_table, state, action, reward, new_state, alpha, gamma, action_space):
    state_key = state_to_key(state)
    new_state_key = state_to_key(new_state)

    if state_key not in q_table:
        q_table[state_key] = np.zeros(action_space)
    if new_state_key not in q_table:
        q_table[new_state_key] = np.zeros(action_space)

    predict = q_table[state_key][action]
    target = reward + gamma * np.max(q_table[new_state_key])
    q_table[state_key][action] += alpha * (target - predict)

def main():
    env = gym.make('SpaceInvaders-v4', render_mode="human")
    action_space = env.action_space.n

    q_table = initialize_q_table()
    alpha = 0.1
    gamma = 0.6
    epsilon = 0.1

    episodes = 10000
    max_steps_per_episode = 100

    for episode in range(episodes):
        state = env.reset()
        terminated = False

        for step in range(max_steps_per_episode):
            env.render()
            action = choose_action(state, q_table, action_space, epsilon)
            new_state, reward, terminated, truncated, info = env.step(action)
            update_q_table(q_table, state, action, reward, new_state, alpha, gamma, action_space)

            state = new_state
            if terminated:
                break

    env.close()

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
