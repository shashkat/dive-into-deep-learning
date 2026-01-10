import random
import numpy as np
import matplotlib.pyplot as plt
seed = 0 # Random number generator seed
random.seed(seed) # Set the random seed to ensure results can be reproduced
np.random.seed(seed)

# function to create an env, and return a dict storing relevant information corresponding to
# the created env.
def frozen_lake(seed, map_name = '4x4'):
    import gymnasium as gym
    env = gym.make('FrozenLake-v1', is_slippery=False, map_name = map_name)
    env.reset(seed = seed) # I believe this applies the random seed to the action space too
    env_unwrapped = env.unwrapped
    env_info = {}
    env_info['desc'] = env_unwrapped.desc  # 2D array specifying what each grid item means
    env_info['num_states'] = env_unwrapped.observation_space.n  # Number of observations/states or obs/state dim
    env_info['num_actions'] = env.action_space.n  # Number of actions or action dim
    # Define indices for (transition probability, nextstate, reward, done) tuple
    env_info['trans_prob_idx'] = 0  # Index of transition probability entry
    env_info['nextstate_idx'] = 1  # Index of next state entry
    env_info['reward_idx'] = 2  # Index of reward entry
    env_info['done_idx'] = 3  # Index of done entry
    env_info['mdp'] = {}
    env_info['env'] = env
    env_unwrapped.P.items()

    # in env_info['mdp'], add entries with (current_state, action) as keys, and corresponding pxrd 
    # (tuple of probability of next state, next state, reward, done or not) as values.
    # s is key and others is value in the dict
    for (s, others) in env_unwrapped.P.items():
        # others(s) = {a0: [ (p(s'|s,a0), s', reward, done),...], a1:[...], ...}

        for (a, pxrds) in others.items():
            # pxrds is [(p1,next1,r1,d1),(p2,next2,r2,d2),..].
            # e.g. [(0.3, 0, 0, False), (0.3, 0, 0, False), (0.3, 4, 1, False)]
            env_info['mdp'][(s,a)] = pxrds

    return (env_info)

# make plots showing the value functions of each state and best action from each state at each 
# timepoint
def show_value_function_progress(env_desc, V, pi):
    """Defined in :numref:`sec_utils`"""
    # This function visualizes how value and policy changes over time.
    # V: [num_iters, num_states]
    # pi: [num_iters, num_states]
    # How to visualize value function is adapted (but changed) from: https://sites.google.com/view/deep-rl-bootcamp/labs

    num_iters = V.shape[0]
    fig, ax  = plt.subplots(figsize=(15, 15))

    for k in range(V.shape[0]):
        plt.subplot(4, 4, k + 1)
        plt.imshow(V[k].reshape(4,4), cmap="bone")
        ax = plt.gca()
        ax.set_xticks(np.arange(0, 5)-.5, minor=True)
        ax.set_yticks(np.arange(0, 5)-.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_xticks([])
        ax.set_yticks([])

        # LEFT action: 0, DOWN action: 1
        # RIGHT action: 2, UP action: 3
        action2dxdy = {0:(-.25, 0),1: (0, .25),
                       2:(0.25, 0),3: (-.25, 0)}

        for y in range(4):
            for x in range(4):
                action = pi[k].reshape(4,4)[y, x]
                dx, dy = action2dxdy[action]

                if env_desc[y,x].decode() == 'H':
                    ax.text(x, y, str(env_desc[y,x].decode()),
                       ha="center", va="center", color="y",
                         size=20, fontweight='bold')

                elif env_desc[y,x].decode() == 'G':
                    ax.text(x, y, str(env_desc[y,x].decode()),
                       ha="center", va="center", color="w",
                         size=20, fontweight='bold')

                else:
                    ax.text(x, y, str(env_desc[y,x].decode()),
                       ha="center", va="center", color="g",
                         size=15, fontweight='bold')

                # No arrow for cells with G and H labels
                if env_desc[y,x].decode() != 'G' and env_desc[y,x].decode() != 'H':
                    ax.arrow(x, y, dx, dy, color='r', head_width=0.2, head_length=0.15)

        ax.set_title("Step = "  + str(k + 1), fontsize=20)

    fig.tight_layout()
    plt.show()

# run the value_iteration algorithm
def value_iteration(env_info, gamma, num_iters):
    env_desc = env_info['desc']  # 2D array shows what each item means
    prob_idx = env_info['trans_prob_idx']
    nextstate_idx = env_info['nextstate_idx']
    reward_idx = env_info['reward_idx']
    num_states = env_info['num_states']
    num_actions = env_info['num_actions']
    mdp = env_info['mdp']

    V  = np.zeros((num_iters + 1, num_states))
    Q  = np.zeros((num_iters + 1, num_states, num_actions))
    pi = np.zeros((num_iters + 1, num_states))

    for k in range(1, num_iters + 1):
        for s in range(num_states):
            for a in range(num_actions):
                # Calculate \sum_{s'} p(s'\mid s,a) [r + \gamma v_k(s')]
                for pxrds in mdp[(s,a)]: # this would have only one iteration, as mdp is a dict, and we are accessing a key of that dict, and each key in a dict is unique
                    # mdp(s,a): [(p1,next1,r1,d1),(p2,next2,r2,d2),..]
                    pr = pxrds[prob_idx]  # p(s'\mid s,a)
                    nextstate = pxrds[nextstate_idx]  # Next state
                    reward = pxrds[reward_idx]  # Reward
                    Q[k,s,a] += pr * (reward + gamma * V[k - 1, nextstate])
            # Record max value and max action
            V[k,s] = np.max(Q[k,s,:]) # this statement is true if we assume that our policy function will necessarily have from each state only one possible action, i.e. prob of one action will be 1 and others will be 0 for any state.
            pi[k,s] = np.argmax(Q[k,s,:]) # this is basically our policy function. However, it doesn't store prob of each action for each state. Rather it just stores the action to be taken from each state, because we assume that we take only 1 action from each state.
    # show_value_function_progress(env_desc, V[:-1], pi[:-1])

    return V # V is enough to know if we have reached the start or not during the backtracking in the learning process. If at the last timepoint, V of starting state is nonzero, then we reached.

gamma = 0.95 # Discount factor
num_iters = 100 # Number of iterations. Keeping a large number, as then in these many, we would have surely reached the start during the backtracking that happens during learning, and then we can figure out how many iterations it took for that to happen.
env_info = frozen_lake(seed, map_name = '8x8')
V = value_iteration(env_info=env_info, gamma=gamma, num_iters=num_iters)

# now, we need to check that at what timepoint, the start state became nonzero
for timepoint in range(V.shape[0]):
    if (V[timepoint,0] != 0):
        print(f"First time starting state's value function nonzero at timepoint {timepoint}.")
        break

# Result:
# First time starting state's value function nonzero at timepoint 14.

# Hence, for the current seed of the frozen lake, it needed 14 steps to be able to reach the 
# start state from the goal state during the backpropagation happening during learning, and that
# corresponds to learning the optimal value function, because if any other state hasn't been 
# reached till now, when the path reaches finally from it to the starting state, it will lead 
# to a lower reward for that path, as more steps were taken for that path, and hence gamma was 
# multiplied more times to the reward (which is 1, only obtained once when the goal is reached).

