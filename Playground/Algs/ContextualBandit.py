import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

class ContextualBandit:
    def __init__(self):
        self.state = 0
        self.bandits = np.array([[0.2, 0, -0.0, -5], [0.1, -5, 1, 0.25], [-5, 5, 5, 5]])
        self.num_bandits = self.bandits.shape[0]
        self.num_actions = self.bandits.shape[1]

    def getBandit(self):
        self.state = np.random.randint(0, len(self.bandits))
        return self.state

    def pullArm(self, action):
        bandit = self.bandits[self.state, action]
        result = np.random.randn(1)
        if result > bandit:
            return 1
        else:
            return -1
        

class Agent:
    def __init__(self, lr, s_size, a_size):
        self.state_in = keras.layers.Input(shape=(1,), dtype=tf.int32)
        state_in_OH = tf.one_hot(self.state_in, depth=s_size)
        output = layers.Dense(a_size, activation='sigmoid', use_bias=False, kernel_initializer='ones')(state_in_OH)
        self.output = tf.reshape(output, [-1])
        self.chosen_action = tf.argmax(self.output)

        self.reward_holder = tf.Variable(initial_value=np.zeros((1,), dtype=np.float32))
        self.action_holder = tf.Variable(initial_value=np.zeros((1,), dtype=np.int32))
        self.responsible_weight = tf.gather(self.output, self.action_holder)
        self.loss = -(tf.math.log(self.responsible_weight) * self.reward_holder)

        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

        with tf.GradientTape(persistent=True) as tape:
          self.update = optimizer.minimize(self.loss, var_list=[output], tape=tape)
          del tape



# Clear any previous TensorFlow graph
tf.compat.v1.reset_default_graph()

cBandit = ContextualBandit()
myAgent = Agent(lr=0.001, s_size=cBandit.num_bandits, a_size=cBandit.num_actions)
weights = tf.compat.v1.trainable_variables()[0]

total_episodes = 10000
total_reward = np.zeros([cBandit.num_bandits, cBandit.num_actions])
e = 0.1

# Launch the TensorFlow graph
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    i = 0
    while i < total_episodes:
        s = cBandit.getBandit()

        if np.random.rand(1) < e:
            action = np.random.randint(cBandit.num_actions)
        else:
            action = sess.run(myAgent.chosen_action, feed_dict={myAgent.state_in: [s]})[0]

        reward = cBandit.pullArm(action)

        feed_dict = {myAgent.reward_holder: [reward], myAgent.action_holder: [action], myAgent.state_in: [s]}
        _ = sess.run([myAgent.update, weights], feed_dict=feed_dict)

        total_reward[s, action] += reward
        if i % 500 == 0:
            print("Mean reward for each of the " + str(cBandit.num_bandits) + " bandits: " + str(np.mean(total_reward, axis=1)))
        i += 1

    for a in range(cBandit.num_bandits):
        print("The agent thinks action " + str(np.argmax(sess.run(weights)[a]) + 1) + " for bandit " + str(a + 1) + " is the most promising....")
        if np.argmax(sess.run(weights)[a]) == np.argmin(cBandit.bandits[a]):
            print("...and it was right!")
        else:
            print("...and it was wrong!")