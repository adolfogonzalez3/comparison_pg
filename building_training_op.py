#########################################################################################
# gabrielgarza / openai-gym-policy-gradient 
# policy_gradient.py
## Adolfo's Notes: Policy Update
## R = discounted_episode_rewards_norm
## H(logits, labels) = softmax_cross_entropy_with_logits(logits=logits, labels=labels)
## H(logits, labels) = - summation{ labels * log(logits) }
## loss = mean{ H(logits, labels) * R }
### line 159
with tf.name_scope('loss'):
    neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    loss = tf.reduce_mean(neg_log_prob * self.discounted_episode_rewards_norm)  # reward guided loss

with tf.name_scope('train'):
    self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

#########################################################################################
# dennybritz / reinforcement-learning / PolicyGradient
# CliffWalk REINFORCE with Baseline Solution.ipynb
## Adolfo's Notes: Policy Update
## R = self.target
## p(a) = probability of doing action a
## loss = -log(p(a)) * R
## Simplest out of three; No Baseline
### window 9
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
## Adolfo's Notes: Policy Update
## advantage * e^(log(p(i)) + log(p(i-1)))
## Uses a baseline to reduce variance; 
## Uses a past policy network similar to DQN
## Uses ideas from http://rll.berkeley.edu/deeprlcoursesp17/docs/lec2.pdf
## Specifically slides 17, 18
### line 32
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