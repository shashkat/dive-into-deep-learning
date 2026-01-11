import random
import numpy as np
import matplotlib.pyplot as plt

seed = 0  # Random number generator seed
random.seed(seed)  # Set the random seed
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

# visualize the progress of the q-learning algo with time.
def show_Q_function_progress(env_desc, V_all, pi_all, size):

    """Defined in :numref:`sec_utils`"""
    # This function visualizes how value and policy changes over time.
    # V: [num_iters, num_states]
    # pi: [num_iters, num_states]

    from matplotlib import pyplot as plt
    if size == "4x4":
        side = 4
    if size == "8x8":
        side = 8
    # We want to only shows few values
    num_iters_all = V_all.shape[0]
    num_iters = num_iters_all // 10
    vis_indx = np.arange(0, num_iters_all, num_iters).tolist()
    vis_indx.append(num_iters_all - 1)
    V = np.zeros((len(vis_indx), V_all.shape[1]))
    pi = np.zeros((len(vis_indx), V_all.shape[1]))
    for c, i in enumerate(vis_indx):
        V[c]  = V_all[i]
        pi[c] = pi_all[i]
    num_iters = V.shape[0]
    fig, ax = plt.subplots(figsize=(15, 15))
    for k in range(V.shape[0]):
        plt.subplot(4, 4, k + 1)
        plt.imshow(V[k].reshape(side, side), cmap="bone")
        ax = plt.gca()
        ax.set_xticks(np.arange(0, side+1)-.5, minor=True)
        ax.set_yticks(np.arange(0, side+1)-.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_xticks([])
        ax.set_yticks([])

        # LEFT action: 0, DOWN action: 1
        # RIGHT action: 2, UP action: 3
        action2dxdy = {0:(-.25, 0),1:(0, .25),
                       2:(0.25, 0),3:(-.25, 0)}

        for y in range(side):
            for x in range(side):
                action = pi[k].reshape(side, side)[y, x]
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
        ax.set_title("Step = "  + str(vis_indx[k] + 1), fontsize=20)
    fig.tight_layout()
    plt.show()

# given the current state of Q table (approximation of action-value function for each state),
# and the current state, take the decision to explore or exploit (choose the best action from 
# that state according to current best knowledge (from Q table)).
# env is just used to get the action space from which to sample, if we decide to explore.
def e_greedy(env, Q, s, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample() # returns a random number from {0,1,2,3}

    else:
        return np.argmax(Q[s,:])

# run the q learning algorithm
def q_learning(env_info, gamma, num_iters, alpha, epsilon, size):
    env_desc = env_info['desc']  # 2D array specifying what each grid item means
    env = env_info['env']  # 2D array specifying what each grid item means
    num_states = env_info['num_states']
    num_actions = env_info['num_actions']

    Q  = np.zeros((num_states, num_actions)) # this is probably what they call the lookup table. This was also present in the value-iteration algorithm implementation in the previous chapter, but that had an extra dimension of iterations
    V  = np.zeros((num_iters + 1, num_states)) # value function for each state. Value function is an estimate of the average return over all trajectories that can be taken, with the current policy starting from a particular state.
    pi = np.zeros((num_iters + 1, num_states)) # our policy of taking actions given current state

    for k in range(1, num_iters + 1):
        # Reset environment
        state, _ = env.reset() # underscored thing is extra info about the reset, and looks something like: {'prob': 1}
        done = False
        while not done:
            # Select an action for a given state and act in env based on selected action
            action = e_greedy(env, Q, state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # use the information from terminated (ended prematurely due to completion or error)
            # and truncated to make the done variable true.
            if terminated or truncated: done = True

            # Q-update:
            y = reward + gamma * np.max(Q[next_state,:]) # what the Q value of current state-action should be, given the q values of state-action pairs for the next state.
            
            # correct the q value of current state, making use of the information of what the Q 
            # value of current state-action should be, given the q values of state-action pairs 
            # for the next state. We are basically doing: q = q - a*d(y-q)^2/dq
            Q[state, action] = Q[state, action] + alpha * (y - Q[state, action])

            # Move to the next state
            state = next_state
        # Record max value and max action for visualization purpose only
        for s in range(num_states):
            V[k,s]  = np.max(Q[s,:])
            pi[k,s] = np.argmax(Q[s,:])
    show_Q_function_progress(env_desc, V[:-1], pi[:-1], size = size)

# Now set up the hyperparameters and environment
gamma = 0.95  # Discount factor
num_iters = 5000  # Number of iterations
alpha   = 0.9  # Learning rate
epsilon = 0.9  # Epsilon in epsilion gready algorithm
map_name = '8x8'
env_info = frozen_lake(seed, map_name)
q_learning(env_info=env_info, gamma=gamma, num_iters=num_iters, alpha=alpha, epsilon=epsilon, size = map_name)

# CONCLUSION: So around 4000 iterations the model was able to decipher the correct decisions
# at each state to be able to reach the goal from the start.

