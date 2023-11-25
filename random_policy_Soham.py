import gym
from ale_py import ALEInterface

'''
Our fixed policy is just randomly and uniformly
choosing an action from the action space.
'''
def random_policy(action_space):
    return action_space.sample()

def main():
    ale = ALEInterface()
    ale.loadROM('roms/SpaceInvaders.bin')
    env = gym.make('SpaceInvaders-v4', render_mode="human")
    observation = env.reset()
    terminated = False
    max_reward = 0.0

    while not terminated:
        env.render()
        action = random_policy(env.action_space)
        observation, reward, terminated, truncated, info = env.step(action)
        max_reward = max(max_reward, reward)
        print(max_reward)

        if terminated:
            observation = env.reset()

    env.close()

if __name__ == "__main__":
    main()