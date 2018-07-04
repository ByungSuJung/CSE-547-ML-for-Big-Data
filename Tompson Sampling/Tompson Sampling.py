#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 17:03:58 2018

@author: bangyc
"""


import numpy as np
import matplotlib.pyplot as plt

np.random.seed(618)

# number of arm
K = 5

# probability distribution
true_prob = [1/6, 1/2, 2/3, 3/4, 5/6]
beta_prior_alpha1 = [1, 1, 1, 1, 1]
beta_prior_alpha2 = [1, 1, 1, 1, 1]

beta_post_alpha1 = beta_prior_alpha1
beta_post_alpha2 = beta_prior_alpha2

# total times
T = 666

rewards = []
regrets = []
avg_regrets = []

N = [0, 0, 0, 0, 0]
average_n = []

for t in np.arange(T):
    sampled_post = []
    for k in np.arange(K):
        sampled = np.random.beta(beta_post_alpha1[k], beta_post_alpha2[k])
        sampled_post.append(sampled)
    action = np.argmax(sampled_post)

    # pulled Arm to get reward
    reward = np.random.binomial(1, true_prob[action])
    regret = 5/6 - reward
    N[action] = N[action] + 1

    average_n.append(list(map(lambda x: x / (t + 1), N)))
    rewards.append(reward)
    regrets.append(regret)
    avg_regrets.append(np.mean(regrets))

    # update the posterior
    beta_post_alpha1[action] = beta_post_alpha1[action] + reward
    beta_post_alpha2[action] = beta_post_alpha2[action] + 1 - reward
    
count = 0
for t in np.arange(T):
    if average_n[t][4] > 0.95:
        count += 1
        if count >= 10:
            print("the first time achieve 0.95 and stays 10 steps: %d" % t)
            break
    else:
        count = 0

average_n = np.array(average_n)
estimate_prob = list(map(lambda x, y: x / (x + y), beta_post_alpha1, beta_post_alpha2))
estimate_error = list(map(lambda x, y: 2 * np.sqrt(x*y/((x + y + 1) * (x + y)**2)), beta_post_alpha1, beta_post_alpha2))
# plot average regret vs time

plt.figure(1)
# plt.title("average regret vs time")
plt.xlabel("time")
plt.ylabel("average regret")

plt.plot(np.arange(1, T + 1), avg_regrets)
plt.savefig('q3.png')
plt.close()

# plot true mu, estimate mu and its confidence interval vs arm k
plt.figure(2)
# plt.title("mu vs arm K")
plt.xlabel("K")
plt.ylabel("Mu")

plt.scatter(range(1, 6), true_prob, c='orange')
plt.errorbar(range(1, 6), estimate_prob, yerr=estimate_error, fmt='o', c='blue')
plt.savefig('q4.png')
plt.close()

# plot
plt.figure(3)
# plt.title("average number vs time")
index = np.arange(1, T+1)
a1, = plt.plot(index, average_n[:,0])
a2, = plt.plot(index, average_n[:,1])
a3, = plt.plot(index, average_n[:,2])
a4, = plt.plot(index, average_n[:,3])
a5, = plt.plot(index, average_n[:,4])

plt.legend([a1, a2, a3, a4, a5], ['k=1', 'k=2', 'k=3', 'k=4', 'k=5'])
plt.savefig('q5.png')
plt.close()