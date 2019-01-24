#########################################################################################
# gabrielgarza / openai-gym-policy-gradient 
# policy_gradient.py
# line # 71
def choose_action(self, observation):
    """
        Choose action based on observation
        Arguments:
            observation: array of state, has shape (num_features)
        Returns: index of action we want to choose
    """
    # Reshape observation to (num_features, 1)
    observation = observation[:, np.newaxis]

    # Run forward propagation to get softmax probabilities
    prob_weights = self.sess.run(self.outputs_softmax, feed_dict = {self.X: observation})

    # Select action using a biased sample
    # this will return the index of the action we've sampled
    action = np.random.choice(range(len(prob_weights.ravel())), p=prob_weights.ravel())
return action

#########################################################################################
# dennybritz / reinforcement-learning / PolicyGradient
# CliffWalk REINFORCE with Baseline Solution.ipynb
########################################
# window 9
def predict(self, state, sess=None):
    sess = sess or tf.get_default_session()
    return sess.run(self.action_probs, { self.state: state })

########################################
# window 11
# Take a step
action_probs = estimator_policy.predict(state)
action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
next_state, reward, done, _ = env.step(action)

#########################################################################################
# Ashboy64 / rl-reimplementations
# line 87
def choose_action(self, s):
    s = s[np.newaxis, :]
    a = self.sess.run(self.sample_op, {self.states_placeholder: s})
return a