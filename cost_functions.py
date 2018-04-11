import numpy as np


#========================================================
# 
# Environment-specific cost functions:
#

def cheetah_cost_fn(state, action, next_state):

    score = 0

    score -= (next_state[17] - state[17]) / 0.01 - 0.1 * (np.sum(action**2))



    return score

#========================================================
# 
# Cost function for a whole trajectory:
#

def trajectory_cost_fn(cost_fn, states, actions, next_states):
    trajectory_cost = 0
    for i in range(len(actions)):
        trajectory_cost += cost_fn(states[i], actions[i], next_states[i])
    return trajectory_cost