# Shray Alag
import gym
import numpy as np
from ale_py import ALEInterface

STEPS = 1000
EPSILON = 0.1  # Exploration-exploitation trade-off
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99

def epsilon_greedy_policy(Q, state):
    if np.random.rand() < EPSILON:
        return np.random.randint(Q.shape[1])  # Explore
    else:
        return np.argmax(Q[state])  # Exploit

def main():
    ale = ALEInterface()
    ale.loadROM('roms/SpaceInvaders.bin')
    env = gym.make('SpaceInvaders-v4', render_mode="human")
    observation = env.reset()

    # Initialize Q-table
    action_space_size = env.action_space.n
    observation_space_size = env.observation_space.shape[0]
    Q = np.zeros((observation_space_size, action_space_size))

    for episode in range(STEPS):
        observation = env.reset()
        terminated = False

        while not terminated:
            env.render()

            # Choose action using epsilon-greedy policy
            action = epsilon_greedy_policy(Q, observation)

            # Take the chosen action
            next_observation, reward, terminated, truncated, info = env.step(action)

            # Update Q-value using Q-learning equation
            Q[observation, action] = (1 - LEARNING_RATE) * Q[observation, action] + \
                                     LEARNING_RATE * (reward + DISCOUNT_FACTOR * np.max(Q[next_observation]))

            observation = next_observation

            if terminated:
                print(f"Episode {episode + 1} completed with total reward: {np.sum(Q)}")

    env.close()

if __name__ == "__main__":
    main()


class GymMDP(MDP):
    def __init__(self, env, max_speed: Optional[float] = None, discount: float = 0.99, timeLimit: Optional[int] = None):
        ale = ALEInterface()
        ale.loadROM('roms/SpaceInvaders.bin')
        env = gym.make('SpaceInvaders-v4', render_mode="human")
        self.max_speed = max_speed
        self._time_limit = 10000
        self._discount = discount
        self._actions = list(range(self.env.action_space.n))
        self._reset_seed_gen = np.random.default_rng(0)

    # Return the number of steps before the MDP should be reset.
    @property
    def timeLimit(self) -> int:
        return self._time_limit

    # Return set of actions possible at every state.
    @property
    def actions(self) -> List[ActionT]:
        return self._actions

    # Return the MDP's discount factor
    @property
    def discount(self):
        return self._discount

    # Returns the start state.
    def startState(self): raise NotImplementedError("Override me")

    # Returns a tuple of (nextState, reward, terminated)
    def transition(self, action): raise NotImplementedError("Override me")

class DiscreteGymMDP(GymMDP):

    def __init__(self, env, feature_bins: Union[int, List[int]] = 10, low: Optional[List[float]] = None, high: Optional[List[float]] = None, **kwargs):
        super().__init__(env, **kwargs)
        assert isinstance(self.env.observation_space, gym.spaces.Box) and len(self.env.observation_space.shape) == 1
        low = self.env.observation_space.low if low is None else low
        high = self.env.observation_space.high if high is None else high
        # Convert the environment to a discretized version
        self.bins = create_bins(low, high, feature_bins)

    def startState(self):
        state, _ = self.env.reset(seed=int(self._reset_seed_gen.integers(0, 1e6)))
        return discretize(state, self.bins)

    def transition(self, action):
        nextState, reward, terminal, _, _ = self.env.step(action)
        nextState = discretize(nextState, self.bins)
        return (nextState, reward, terminal)

def simulate(mdp: MDP, rl: RLAlgorithm, numTrials=10, train=True, verbose=False, demo=False):
    
    ale = ALEInterface()
    ale.loadROM('roms/SpaceInvaders.bin')
    env = gym.make('SpaceInvaders-v4', render_mode="human")

    totalRewards = []  # The discounted rewards we get on each trial
    for trial in range(numTrials):
        observation = env.reset()
        state = mdp.startState()

        if demo:
            mdp.env.render()
        totalDiscount = 1
        totalReward = 0
        trialLength = 0
        for _ in range(mdp.timeLimit):
            if demo:
                time.sleep(0.05)
            action = rl.getAction(state, explore=train)
            if action is None: 
                break
            nextState, reward, terminal = mdp.transition(action)
            trialLength += 1
            if train:
                rl.incorporateFeedback(state, action, reward, nextState, terminal)
            
            totalReward += totalDiscount * reward
            totalDiscount *= mdp.discount
            state = nextState

            if terminal:
                break # We have reached a terminal state

        if verbose:
            print(("Trial %d (totalReward = %s, Length = %s)" % (trial, totalReward, trialLength)))
        totalRewards.append(totalReward)
    return totalRewards




class TabularQLearning(util.RLAlgorithm):
    def __init__(self, actions: List[ActionT], discount: float,
                 explorationProb: float = 0.2, initialQ: float = 0):
        self.actions = actions
        self.discount = discount
        self.explorationProb = explorationProb
        self.Q = defaultdict(lambda: initialQ)
        self.numIters = 0

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability |explorationProb|, take a random action.
    # The input boolean |explore| indicates whether the RL algorithm is in train or test time. If it is false (test), we
    # should always choose the maximum Q-value action.
    def getAction(self, state: StateT, explore: bool = True) -> ActionT:
        if explore:
            self.numIters += 1
        explorationProb = self.explorationProb
        if self.numIters < 2e4:  # explore
            explorationProb = 1.0
        elif self.numIters > 1e5:  # Lower the exploration probability by a logarithmic factor.
            explorationProb = explorationProb / \
                math.log(self.numIters - 100000 + 1)

        # BEGIN_YOUR_CODE (our solution is 5 lines of code, but don't worry if you deviate from this)
        population = [0, 1]
        weights = [explorationProb, 1 - explorationProb]
        if (explore == True and random.choices(population, weights)[0] == 0):
            return random.choice(self.actions)

        bestActionValue = None
        bestAction = None
        for action in self.actions:
            curQ = self.Q[(state, action)]
            if bestActionValue is None or curQ >= bestActionValue:
                bestActionValue = curQ
                bestAction = action
        if bestActionValue is not None:
            return bestAction
        return random.choice(self.actions)
        # END_YOUR_CODE

    # Call this function to get the step size to update the weights.
    def getStepSize(self) -> float:
        return 0.1

    # We will call this function with (s, a, r, s'), which you should use to update |Q|.
    # Note that if s is a terminal state, then terminal will be True.  Remember to check for this.
    # You should update the Q values using self.getStepSize()
    def incorporateFeedback(self, state: StateT, action: ActionT, reward: float, nextState: StateT, terminal: bool) -> None:

        # BEGIN_YOUR_CODE (our solution is 8 lines of code, but don't worry if you deviate from this)
        if not terminal:
            self.Q[(state, action)] = (1 - self.getStepSize()) * self.Q[(state, action)] + self.getStepSize() * \
                (reward + self.discount *
                 max(self.Q[nextState, action] for action in self.actions))
        else:
            self.Q[(state, action)] = (1 - self.getStepSize()) * \
                self.Q[(state, action)] + self.getStepSize() * reward
        # END_YOUR_CODE