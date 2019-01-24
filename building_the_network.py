#########################################################################################
# gabrielgarza / openai-gym-policy-gradient 
# policy_gradient.py
# line # 100
def build_network(self):
    with tf.name_scope('inputs'):
        self.X = tf.placeholder(tf.float32, [None, self.n_x], name="X")
        self.Y = tf.placeholder(tf.int32, [None, ], name="Y")
        self.discounted_episode_rewards_norm = tf.placeholder(tf.float32, [None, ], name="actions_value")
    # fc1
    A1 = tf.layers.dense(
        inputs=self.X,
        units=10,
        activation=tf.nn.relu,
        kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
        bias_initializer=tf.constant_initializer(0.1),
        name='fc1'
    )
    # fc2
    A2 = tf.layers.dense(
        inputs=A1,
        units=10,
        activation=tf.nn.relu,
        kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
        bias_initializer=tf.constant_initializer(0.1),
        name='fc2'
    )
    # fc3
    Z3 = tf.layers.dense(
        inputs=A2,
        units=self.n_y,
        activation=None,
        kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
        bias_initializer=tf.constant_initializer(0.1),
        name='fc3'
    )

    # Softmax outputs
    self.outputs_softmax = tf.nn.softmax(Z3, name='A3')

#########################################################################################
# dennybritz / reinforcement-learning / PolicyGradient
# CliffWalk REINFORCE with Baseline Solution.ipynb
########################################
# window 9
with tf.variable_scope(scope):
    self.state = tf.placeholder(tf.int32, [], "state")
    self.action = tf.placeholder(dtype=tf.int32, name="action")
    self.target = tf.placeholder(dtype=tf.float32, name="target")

    # This is just table lookup estimator
    state_one_hot = tf.one_hot(self.state, int(env.observation_space.n))
    self.output_layer = tf.contrib.layers.fully_connected(
        inputs=tf.expand_dims(state_one_hot, 0),
        num_outputs=env.action_space.n,
        activation_fn=None,
        weights_initializer=tf.zeros_initializer)

########################################
# window 10
with tf.variable_scope(scope):
    self.state = tf.placeholder(tf.int32, [], "state")
    self.target = tf.placeholder(dtype=tf.float32, name="target")

    # This is just table lookup estimator
    state_one_hot = tf.one_hot(self.state, int(env.observation_space.n))
    self.output_layer = tf.contrib.layers.fully_connected(
        inputs=tf.expand_dims(state_one_hot, 0),
        num_outputs=1,
        activation_fn=None,
        weights_initializer=tf.zeros_initializer)


#########################################################################################
# Ashboy64 / rl-implementations / Reimplementations / Vanilla-Policy-Gradient
# vanilla_pg.py
# line # 72
def build_policy(self, name, trainable):
    with tf.variable_scope(name):
        l1 = tf.layers.dense(self.states_placeholder, 100, tf.nn.relu, trainable=trainable)
        l2 = tf.layers.dense(l1, 100, tf.nn.relu, trainable=trainable)
        pd = Probability_Distribution(tf.layers.dense(l2, self.env.action_space.n, trainable=trainable))
    params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
    # return norm_dist, params
    return pd, params

def build_critic(self):
    with tf.variable_scope('critic'):
        l1 = tf.layers.dense(self.states_placeholder, 100, tf.nn.relu)
        val = tf.layers.dense(l1, 1)
return val
