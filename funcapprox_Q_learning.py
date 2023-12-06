import gym
import numpy as np
import random
import math
from skimage.transform import resize


class FunctionApproxQLearning:
    def __init__(self, feature_dim, feature_extractor, actions, discount, exploration_prob=0.2):
        self.feature_dim = feature_dim
        self.feature_extractor = feature_extractor
        self.actions = actions
        self.discount = discount
        self.exploration_prob = exploration_prob
        self.W = np.random.standard_normal(size=(feature_dim, len(actions)))
        self.num_iters = 0

    def getQ(self, state, action):
        features = self.feature_extractor(state)
        q_value_state = np.matmul(features, self.W)
        return q_value_state[action]

    def getAction(self, state, explore=True):
        self.num_iters += 1
        exploration_prob = self.exploration_prob
        if self.num_iters < 20000:  # Always explore
            exploration_prob = 1.0
        elif self.num_iters > 100000:  # Lower exploration probability
            exploration_prob = exploration_prob / math.log(self.num_iters - 100000 + 1)

        if explore and random.random() < exploration_prob:
            return random.choice(self.actions)
        else:
            q_values = [self.getQ(state, action) for action in self.actions]
            return self.actions[np.argmax(q_values)]

    def getStepSize(self):
        return 0.005 * (0.99) ** (self.num_iters / 500)

    def incorporateFeedback(self, state, action, reward, next_state, terminal):
        if terminal:
            future_value = 0
        else:
            future_value = max(self.getQ(next_state, new_action) for new_action in self.actions)

        old_value = self.getQ(state, action)
        new_value = reward + self.discount * future_value
        step_size = self.getStepSize()
        features = self.feature_extractor(state)
        self.W[:, action] += step_size * (new_value - old_value) * features

def feature_extractor(state):
    if isinstance(state, tuple):
        image_data = state[0]
    else:
        image_data = state
    resized_image = resize(image_data, output_shape=(10, 10), anti_aliasing=True, mode='constant')
    if len(resized_image.shape) > 2 and resized_image.shape[2] == 3:
        resized_image = np.dot(resized_image[..., :3], [0.2989, 0.5870, 0.1140])
    return resized_image.flatten()


def main():
    env = gym.make('SpaceInvaders-v4', render_mode="human")
    feature_dim = 100  
    actions = list(range(env.action_space.n))
    discount = 0.99
    q_learning = FunctionApproxQLearning(feature_dim, feature_extractor, actions, discount)

    episodes = 300

    for episode in range(episodes):
        state = env.reset()
        terminated = False
        total_reward = 0  

        while not terminated:
            env.render()
            action = q_learning.getAction(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            q_learning.incorporateFeedback(state, action, reward, next_state, terminated)
            state = next_state
            total_reward += reward
        
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")  # Print the total reward for the episode

    env.close()

if __name__ == "__main__":
    main()
