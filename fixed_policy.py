import gym
import random
from ale_py import ALEInterface
# from ale_py.roms import SpaceInvaders

STEPS = 1000

def main():
    ale = ALEInterface()
    ale.loadROM('roms/SpaceInvaders.bin')
    env = gym.make('SpaceInvaders-v4', render_mode="human")
    observation = env.reset()
    terminated = False
    
    action_numb = random.randint(0, 5)

    while not terminated:  # You can adjust the number of steps here
        env.render()
        # print(env.action_space.sample())
        action = action_numb # Replace with your agent's action selection
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated:
            observation = env.reset()

    env.close()

if __name__ == "__main__":
    main()

# import tensorflow as tf      # Deep Learning library
# import numpy as np           # Handle matrices
# import retro                 # Retro Environment


# from skimage import transform # Help us to preprocess the frames
# from skimage.color import rgb2gray # Help us to gray our frames

# import matplotlib.pyplot as plt # Display graphs

# from collections import deque# Ordered collection with ends

# import random

# import warnings # This ignore all the warning messages that are normally printed during the training because of skiimage
# warnings.filterwarnings('ignore')

# """
# preprocess_frame:
# Take a frame.
# Grayscale it
# Resize it.
#     __________________
#     |                 |
#     |                 |
#     |                 |
#     |                 |
#     |_________________|
    
#     to
#     _____________
#     |            |
#     |            |
#     |            |
#     |____________|
# Normalize it.

# return preprocessed_frame

# """

# def preprocess_frame(frame):
#     # Greyscale frame 
#     gray = rgb2gray(frame)
    
#     # Crop the screen (remove the part below the player)
#     # [Up: Down, Left: right]
#     cropped_frame = gray[8:-12,4:-12]
    
#     # Normalize Pixel Values
#     normalized_frame = cropped_frame/255.0
    
#     # Resize
#     # Thanks to Mikołaj Walkowiak
#     preprocessed_frame = transform.resize(normalized_frame, [110,84])
    
#     return preprocessed_frame # 110x84x1 frame

# stack_size = 4 # We stack 4 frames

# # Initialize deque with zero-images one array for each image
# stacked_frames  =  deque([np.zeros((110,84), dtype=np.int) for i in range(stack_size)], maxlen=4)

# def stack_frames(stacked_frames, state, is_new_episode):
#     # Preprocess frame
#     frame = preprocess_frame(state)
    
#     if is_new_episode:
#         # Clear our stacked_frames
#         stacked_frames = deque([np.zeros((110,84), dtype=np.int) for i in range(stack_size)], maxlen=4)
        
#         # Because we're in a new episode, copy the same frame 4x
#         stacked_frames.append(frame)
#         stacked_frames.append(frame)
#         stacked_frames.append(frame)
#         stacked_frames.append(frame)
        
#         # Stack the frames
#         stacked_state = np.stack(stacked_frames, axis=2)
        
#     else:
#         # Append frame to deque, automatically removes the oldest frame
#         stacked_frames.append(frame)

#         # Build the stacked state (first dimension specifies different frames)
#         stacked_state = np.stack(stacked_frames, axis=2) 
    
#     return stacked_state, stacked_frames
    

# def main():
#     # Create our environment
#     env = retro.make(game='SpaceInvaders-Atari2600')

#     print("The size of our frame is: ", env.observation_space)
#     print("The action size is : ", env.action_space.n)

#     # Here we create an hot encoded version of our actions
#     # possible_actions = [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0]...]
#     possible_actions = np.array(np.identity(env.action_space.n,dtype=int).tolist())

#     ### MODEL HYPERPARAMETERS
#     state_size = [110, 84, 4]      # Our input is a stack of 4 frames hence 110x84x4 (Width, height, channels) 
#     action_size = env.action_space.n # 8 possible actions
#     learning_rate =  0.00025      # Alpha (aka learning rate)

#     ### TRAINING HYPERPARAMETERS
#     total_episodes = 50            # Total episodes for training
#     max_steps = 50000              # Max possible steps in an episode
#     batch_size = 64                # Batch size

#     # Exploration parameters for epsilon greedy strategy
#     explore_start = 1.0            # exploration probability at start
#     explore_stop = 0.01            # minimum exploration probability 
#     decay_rate = 0.00001           # exponential decay rate for exploration prob

#     # Q learning hyperparameters
#     gamma = 0.9                    # Discounting rate

#     ### MEMORY HYPERPARAMETERS
#     pretrain_length = batch_size   # Number of experiences stored in the Memory when initialized for the first time
#     memory_size = 1000000          # Number of experiences the Memory can keep

#     ### PREPROCESSING HYPERPARAMETERS
#     stack_size = 4                 # Number of frames stacked

#     ### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
#     training = False

#     ## TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
#     episode_render = False

# if __name__ == "__main__":
#     main()