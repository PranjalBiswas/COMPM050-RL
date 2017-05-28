import numpy as np
import pprint
import sys
if "../" not in sys.path:
  sys.path.append("../")
from lib.envs.gridworld import GridworldEnv

pp = pprint.PrettyPrinter(indent=2)
env = GridworldEnv()

def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.

    Args:
        env: OpenAI environment. env.P represents the transition probabilities
        of the environment.

        theta: Stopping threshold. If the value of all states changes less
        than theta in one iteration we are done.

        discount_factor: lambda time discount factor.

    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value
        function.
    """

    V = np.zeros(env.nS)
    policy = np.zeros([env.nS, env.nA])

    # Implement!

    # Start with Random Policy
    v = np.zeros(env.nS)
    policy = np.ones([env.nS, env.nA])/env.nA

    while True:
        v = np.zeros(env.nS)
        # One step policy evaluation
        for s in range(env.nS):
            a = np.argmax(policy[s])
            for transition_prob, next_state, reward, done in env.P[s][a]:
                v[s] += reward + discount_factor * transition_prob * V[next_state]
        # Policy update (Greedy)
        print(V.reshape(4,4))
        for s in range(env.nS):
            action_values = np.zeros(env.nA)
            for a in range(env.nA):
                for transition_prob, next_state, reward, done in env.P[s][a]:
                    action_values[a] += reward + transition_prob * v[next_state]
            greedy_action = np.argmax(action_values)
            policy[s] = np.eye(env.nA)[greedy_action]
        print(policy.reshape(16, 4))
        # Check Convergency
        if max(V-v) < theta:
            break
        V = v

    return policy, V

policy, v = value_iteration(env)

print("Policy Probability Distribution:")
print(policy)
print("")

print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(policy, axis=1), env.shape))
print("")

print("Value Function:")
print(v)
print("")

print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
print("")


# Test the value function
expected_v = np.array([ 0, -1, -2, -3, -1, -2, -3, -2, -2, -3, -2, -1, -3, -2, -1,  0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)
