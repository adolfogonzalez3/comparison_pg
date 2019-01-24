#########################################################################################
# gabrielgarza / openai-gym-policy-gradient 
# policy_gradient.py
# line # 159
with tf.name_scope('loss'):
    neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=Z3, labels=self.Y)
    loss = tf.reduce_mean(neg_log_prob * self.discounted_episode_rewards_norm)  # reward guided loss

with tf.name_scope('train'):
    self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

#########################################################################################
# dennybritz / reinforcement-learning / PolicyGradient
# CliffWalk REINFORCE with Baseline Solution.ipynb
# window 9
self.action_probs = tf.squeeze(tf.nn.softmax(self.output_layer))
self.picked_action_prob = tf.gather(self.action_probs, self.action)

# Loss and train op
self.loss = -tf.log(self.picked_action_prob) * self.target

self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
self.train_op = self.optimizer.minimize(
    self.loss, global_step=tf.contrib.framework.get_global_step())

# Calculate the loss
self.losses = tf.squared_difference(self.y_pl, self.action_predictions)
self.loss = tf.reduce_mean(self.losses)

# Optimizer Parameters from original paper
self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

########################################
# window 10
self.value_estimate = tf.squeeze(self.output_layer)
self.loss = tf.squared_difference(self.value_estimate, self.target)

self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
self.train_op = self.optimizer.minimize(
self.loss, global_step=tf.contrib.framework.get_global_step())

#########################################################################################
# Ashboy64 / rl-reimplementations
# line 32
# critic
self.value = self.build_critic()
self.advantage = self.dicounted_rewards_placeholder - self.value
self.closs = tf.reduce_mean(tf.square(self.advantage))
self.ctrain_op = tf.train.AdamOptimizer(self.c_lr).minimize(self.closs)

# actor
pi, pi_params = self.build_policy('pi', trainable=True)
oldpi, oldpi_params = self.build_policy('oldpi', trainable=False)
with tf.variable_scope('sample_action'):
    self.sample_op = tf.squeeze(pi.sample(), axis=0)
with tf.variable_scope('update_oldpi'):
    self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

with tf.variable_scope('loss'):
    with tf.variable_scope('surrogate'):
        ratio = tf.exp(pi.logp(self.actions_placeholder) - oldpi.logp(self.actions_placeholder))
        surr = ratio * self.advantages_placeholder
    self.aloss = -tf.reduce_mean(surr)

with tf.variable_scope('atrain'):
self.atrain_op = tf.train.AdamOptimizer(self.a_lr).minimize(self.aloss)