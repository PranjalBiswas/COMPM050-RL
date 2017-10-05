import numpy as np
import tensorflow as tf
import random
import gym
import itertools
import math
import matplotlib.pyplot as plt
import matplotlib
import sys
import collections
# import ipdb

if "../" not in sys.path:
  sys.path.append("../")
from lib import plotting
matplotlib.style.use('ggplot')


is_submit = False
api_key = 'sk_HZdo670QLuzq21Do3P0sg'

env = gym.make('CartPole-v0')
env = gym.wrappers.Monitor(env, 'results', force=True)

class PolicyEstimator():
    """Policy Function Approximator"""

    def __init__(self, learning_rate=0.01, scope="policy_estimator"):
        """
        Input Layer(4) - Output Layer(2, softmax)
        VS
        Input Layer(4) - Output Layer(2, softmax)
        """
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32, [4])
            self.action = tf.placeholder(tf.int32, [])
            self.target = tf.placeholder(tf.float32, [])
            self.output_layer = tf.contrib.layers.fully_connected(
                    inputs=tf.expand_dims(self.state, 0),
                    num_outputs=env.action_space.n,
                    activation_fn=tf.nn.softmax,
                    weights_initializer=tf.zeros_initializer)
            self.action_probs = tf.squeeze(tf.nn.softmax(self.output_layer))
            self.picked_action_prob = tf.gather(self.action_probs, self.action)
            self.loss = -tf.log(self.picked_action_prob)*self.target
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                    self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.action_probs, { self.state:state })

    def update(self, state, target, action, sess=None):
        sess = sess or tf.get_default_session()
        _, loss = sess.run([self.train_op, self.loss],
                { self.state:state, self.target:target, self.action:action })
        return loss

class ValueEstimator():
    """Value Function Approximator
    Input Layer(4) - Output Layer(1, ReLU)
    VS
    Input Layer(4) - Hidden Layer(2, ReLU) - Output Layer(1, x)
    """

    # def __init__(self, learning_rate=0.1, scope="value_estimator"):
            # with tf.variable_scope(scope):
                # self.state = tf.placeholder(dtype=tf.float32, shape=[4],
                   # name="state")
                # self.target = tf.placeholder(dtype=tf.float32, shape=[],
                   # name="target")
                # self.output_layer = tf.contrib.layers.fully_connected(
                   # inputs=tf.expand_dims(self.state, 0),
                   # num_outputs=1,
                   # activation_fn=tf.nn.relu,
                   # weights_initializer=tf.zeros_initializer)
                # self.value_estimate = tf.squeeze(self.output_layer)
                # self.loss = tf.squared_difference(self.value_estimate, self.target)
                # self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                # self.train_op = self.optimizer.minimize(
                   # self.loss, global_step=tf.contrib.framework.get_global_step())

    def __init__(self, learning_rate=0.1, scope="value_estimator"):
            with tf.variable_scope(scope):
                self.state = tf.placeholder(dtype=tf.float32, shape=[4],
                   name="state")
                self.target = tf.placeholder(dtype=tf.float32, shape=[],
                   name="target")
                self.hidden_layer = tf.contrib.layers.fully_connected(
                   inputs=tf.expand_dims(self.state, 0),
                   num_outputs=2,
                   activation_fn=tf.nn.relu,
                   weights_initializer=tf.zeros_initializer)
                self.output_layer = tf.contrib.layers.fully_connected(
                        inputs=self.hidden_layer,
                        num_outputs=1,
                        activation_fn=None,
                        weights_initializer=tf.zeros_initializer)
                self.value_estimate = tf.squeeze(self.output_layer)
                # self.loss = tf.squared_difference(self.value_estimate, self.target)
                self.loss = tf.nn.l2_loss(self.value_estimate - self.target)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                self.train_op = self.optimizer.minimize(
                   self.loss, global_step=tf.contrib.framework.get_global_step())


    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.value_estimate, { self.state:state })

    def update(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        _, loss = sess.run([self.train_op, self.loss], { self.state:state,
            self.target:target})
        return loss

def actor_critic(env, estimator_policy, estimator_value, num_episodes,
                 discount_factor=1.0):
    """
    Target for Policy Estimator - TD Error
    Target for Value Estimator - TD Target
    VS
    Target for Policy Estimator - MonteCarlo
    Target for Value Estimator - MonteCarlo


    :env: OpenAI Environment
    :estimator_policy: Policy Estimator
    :estimator_value: Value Estimator
    :returns: EpisodeStats object with two numpy arrays for episode_lengths and
              episode_rewards

    """
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    Transition = collections.namedtuple("transition", ["state", "action",
                                                       "reward", "next_state",
                                                       "done"])

    ## TD(0)
    # for i_episode in range(num_episodes):
        # state = env.reset()
        # episode = []

        # for t in itertools.count():
            # Take a Step and Append to Episode
            # action_probs = estimator_policy.predict(state)
            # action = np.random.choice( np.arange(len(action_probs)), p=action_probs)
            # next_state, reward, done, _ = env.step(action)
            # episode.append(Transition( state=state, action=action,
                # reward=reward, next_state=next_state, done=done ))

            # Update stats
            # stats.episode_rewards[i_episode] += reward
            # stats.episode_lengths[i_episode] = t

            # Calc TD Target
            # value_next = estimator_value.predict(next_state)
            # td_target = reward + discount_factor * value_next
            # td_error = td_target - estimator_value.predict(state)

            # Update value estimator, policy estimator
            # estimator_value.update(state, td_target)
            # estimator_policy.update(state, td_error, action)

            # if done:
                # break

            # state = next_state

        # print("\rEpisode {}/{} ({})".format(i_episode+1, num_episodes,
            # stats.episode_rewards[i_episode-1]), end="")

    # return stats

    ## MC
    for i_episode in range(num_episodes):
        state = env.reset()
        episode = []

        for t in itertools.count():
            ## Take a Step and Append to Episode
            action_probs = estimator_policy.predict(state)

            # action = 0 if random.uniform(0, 1) < action_probs[0] else 1
            action = np.random.choice( np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)
            episode.append(Transition( state=state, action=action,
                reward=reward, next_state=next_state, done=done ))

            ## Update stats
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            if done:
                break

            state = next_state

        print("\rEpisode {}/{} ({})".format(i_episode+1, num_episodes,
            stats.episode_rewards[i_episode-1]), end="")

        for t, transition in enumerate(episode):
            total_return = sum(discount_factor**i * trans.reward
                    for i, trans in enumerate(episode[t:]))
            estimator_value.update(transition.state, total_return)
            baseline_value = estimator_value.predict(transition.state)
            advantage = total_return - baseline_value
            estimator_policy.update(transition.state, advantage, transition.action)

    return stats



tf.reset_default_graph()

global_step = tf.Variable(0, name="global_step", trainable=False)
policy_estimator = PolicyEstimator()
value_estimator = ValueEstimator()

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    stats = actor_critic(env, policy_estimator, value_estimator, 1000)


print("\nLast 10 episode : ", np.sum(stats.episode_rewards[-10:])/10)
if is_submit and np.sum(stats.episode_rewards[-10:])/10 > 195:
    print("submitting to gym")
    gym.scoreboard.api_key = api_key
    env.close()
    gym.upload('results')


#plot
# env.close()
# plotting.plot_episode_stats(stats, smoothing_window=10)
