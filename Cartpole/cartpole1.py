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
import ipdb

if "../" not in sys.path:
  sys.path.append("../")
from lib import plotting
matplotlib.style.use('ggplot')


is_submit = False
api_key = 'sk_HZdo670QLuzq21Do3P0sg'

env = gym.make('CartPole-v0')
env = gym.wrappers.Monitor(env, 'results', force=True)

class PolicyEstimator():
    """
    Policy Function Approximator
    """

    def __init__(self, learning_rate=0.1, scope="policy_estimator"):
        with tf.variable_scope(scope):

            self.state = tf.placeholder(tf.float32, [None, 4])
            self.action = tf.placeholder(tf.float32, [None, 2])
            self.target = tf.placeholder(tf.float32, [None, 1])
            self.w1 = tf.get_variable("w1", [4,2])
            self.h1 = tf.nn.softmax(tf.matmul(self.state, self.w1))
            self.action_prob = self.h1
            self.picked_action_prob = tf.reduce_sum(tf.multiply(self.action_prob, self.action), reduction_indices=[1])
            self.loss = tf.reduce_sum(-tf.log(self.picked_action_prob)*self.target)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                    self.loss, global_step=tf.contrib.framework.get_global_step())

            # self.state = tf.placeholder(tf.float32, [None, 4])
            # self.action = tf.placeholder(tf.float32, [None, 2])
            # self.target = tf.placeholder(tf.float32, [None, 1])
            # self.action_prob = tf.contrib.layers.fully_connected(
                # inputs=self.state,
                # num_outputs=env.action_space.n,
                # activation_fn=tf.nn.softmax)
            # self.picked_action_prob = tf.reduce_sum(tf.multiply(self.action_prob, self.action), reduction_indices=[1])
            # self.loss = tf.reduce_sum(-tf.log(self.picked_action_prob)*self.target)
            # self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            # self.train_op = self.optimizer.minimize(
                    # self.loss, global_step=tf.contrib.framework.get_global_step())



    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.action_prob, { self.state:state })

    def update(self, state, target, action, sess=None):
        sess = sess or tf.get_default_session()
        _, loss = sess.run([self.train_op, self.loss],
                { self.state:state, self.target:target, self.action:action })
        return loss

class ValueEstimator():
    """Value Function Approximator
    Input Layer(4) - Hidden Layer(2, ReLU) - Output Layer(1, x)
    """

    def __init__(self, learning_rate=0.1, scope="value_estimator"):
            with tf.variable_scope(scope):

                self.state = tf.placeholder(dtype=tf.float32, shape=[None, 4],
                    name="state")
                self.target = tf.placeholder(dtype=tf.float32, shape=[None, 1],
                    name="target")
                self.w1 = tf.get_variable("w1", [4,2])
                self.b1 = tf.get_variable("b1", [2])
                self.h1 = tf.nn.relu(tf.matmul(self.state, self.w1)+self.b1)
                self.w2 = tf.get_variable("w2", [2,1])
                self.b2 = tf.get_variable("b2", [1])
                self.h2 = tf.matmul(self.h1,self.w2)+self.b2
                self.value_estimate = self.h2
                self.loss = tf.nn.l2_loss(self.value_estimate - self.target)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                self.train_op = self.optimizer.minimize(
                   self.loss, global_step=tf.contrib.framework.get_global_step())


                # self.state = tf.placeholder(dtype=tf.float32, shape=[None, 4],
                    # name="state")
                # self.target = tf.placeholder(dtype=tf.float32, shape=[None, 1],
                    # name="target")
                # self.hidden1 = tf.contrib.layers.fully_connected(
                    # inputs=self.state,
                    # num_outputs=2,
                    # activation_fn=tf.nn.relu)
                # self.value_estimate = tf.contrib.layers.fully_connected(
                    # inputs=self.hidden1,
                    # num_outputs=1,
                    # activation_fn=None)
                # self.loss = tf.nn.l2_loss(self.value_estimate - self.target)
                # self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                # self.train_op = self.optimizer.minimize(
                   # self.loss, global_step=tf.contrib.framework.get_global_step())



    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.value_estimate, { self.state:state })

    def update(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        _, loss = sess.run([self.train_op, self.loss], { self.state:state,
            self.target:target})
        return loss

def actor_critic(env, estimator_policy, estimator_value,  num_episodes,
                 discount_factor=0.99, render=False,):
    """
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

    for i_episode in range(num_episodes):
        state = env.reset()
        episode = []
        state_batch = []
        action_vector_batch = []
        advantage_batch = []
        target_batch = []

        for t in itertools.count():

            ## Take a Step and Append to Episode
            state_batch.append(state)
            state_vector = np.expand_dims(state, axis=0)
            action_prob = estimator_policy.predict(state_vector)
            action = 0 if random.uniform(0,1) < action_prob[0][0] else 1
            action_vector = np.zeros(2)
            action_vector[action] = 1
            action_vector_batch.append(action_vector)
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
            target_batch.append([total_return])
            baseline_value = estimator_value.predict( np.expand_dims(transition.state,0) )[0][0]
            advantage = total_return - baseline_value
            advantage_batch.append([advantage])

        ## update value estimator (state, target(value))
        estimator_value.update(state_batch, target_batch)
        ## update policy estimator (state, target(advantage), action)?
        estimator_policy.update(state_batch, advantage_batch, action_vector_batch)

    return stats



tf.reset_default_graph()

global_step = tf.Variable(0, name="global_step", trainable=False)
policy_estimator = PolicyEstimator()
value_estimator = ValueEstimator()

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    stats = actor_critic(env, policy_estimator, value_estimator, 200)


print("\nLast 50 episode : ", np.sum(stats.episode_rewards[-50:])/50)
if is_submit and np.sum(stats.episode_rewards[-50:])/50 > 195:
    print("submitting to gym")
    gym.scoreboard.api_key = api_key
    env.close()
    gym.upload('results')

#plot
# env.close()
# plotting.plot_episode_stats(stats, smoothing_window=10)
