#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-based Reinforcement Learning policies
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
2021
By Thomas Moerland
"""
from urllib.parse import _NetlocResultMixinStr
import numpy as np
from queue import PriorityQueue
from env import WindyGridworld

class DynaAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states,n_actions))
        # TO DO: Initialize count tables, and reward sum tables. 
        self.counts = np.zeros((n_states, n_actions, n_states))
        self.rewardsum = np.zeros((n_states, n_actions, n_states))
        self.estimate = np.zeros((n_states, n_actions, n_states))
        self.rewestimate = np.zeros((n_states, n_actions, n_states))
        
    def select_action(self, s, epsilon):
        # TO DO: Add own code
        prob = np.random.random()
        if prob < epsilon:
            return np.random.randint(0, self.n_actions)
        else:
            high = np.where(self.Q_sa[s, ] == np.amax(self.Q_sa[s, ]))
            return np.random.choice(high[0])

        
    def update(self,s,a,r,done,s_next,n_planning_updates):        
        # TO DO: Add own code
        actions = []
        obs_states = []
        #update        
        a_next = np.random.choice(np.where(self.Q_sa[s_next, ] == np.amax(self.Q_sa[s_next, ]))[0])
        self.Q_sa[s,a] += self.learning_rate * ( r + self.gamma * self.Q_sa[s_next, a_next ] - self.Q_sa[s,a])

        #make model
        self.counts[s, a, s_next] += 1
        self.rewardsum[s, a, s_next] += r
        self.estimate[ s, a, s_next] = self.counts[s, a, s_next]/np.sum(self.counts[s, a])
        self.rewestimate [s, a, s_next] = self.rewardsum[s, a, s_next]/self.counts[s, a, s_next]

        #find previous states and actions
        for i in range(self.n_states):        #loop through states
            actions.append([])
            nonzerostate = False
            act = []
            for z in range(self.n_actions): #loop through actions 
                nonzero = False
                for j in range(self.n_states): #loop through primes
                    if self.counts[i, z, j]!=0:
                        nonzero = True
                        break
                if nonzero:
                    act.append(z)
                    nonzerostate = True            
            if nonzerostate:                
                actions[i] = act
                obs_states.append(i)


        for i in range(n_planning_updates):
            randstate = np.random.choice(obs_states)
            randaction = np.random.choice(actions[randstate])                                                   
            est_state = np.random.choice(np.where(self.estimate[randstate, randaction] == np.amax(self.estimate[randstate, randaction]))[0])
            est_rew = self.rewestimate[randstate, randaction, est_state]
            a_next = np.random.choice(np.where(self.Q_sa[est_state, ] == np.amax(self.Q_sa[est_state, ]))[0])

            self.Q_sa[randstate, randaction] += self.learning_rate * ( est_rew + self.gamma * self.Q_sa[est_state, a_next] - self.Q_sa[randstate,randaction])


        pass
    
class PrioritizedSweepingAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma, max_queue_size=200, priority_cutoff=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.priority_cutoff = priority_cutoff
        self.queue = PriorityQueue()
        
        self.Q_sa = np.zeros((n_states,n_actions))
        # TO DO: Initialize count tables, and reward sum tables.
        self.counts = np.zeros((n_states, n_actions, n_states))
        self.rewardsum = np.zeros((n_states, n_actions, n_states)) 
        self.estimate = np.zeros((n_states, n_actions, n_states))
        self.rewestimate = np.zeros((n_states, n_actions, n_states))


    def select_action(self, s, epsilon):
        # TO DO: Add own code
        prob = np.random.random()
        if prob < epsilon:
            return np.random.randint(0, self.n_actions)
        else:
            high = np.where(self.Q_sa[s, ] == np.amax(self.Q_sa[s, ]))
            return np.random.choice(high[0])
        
        
    def update(self,s,a,r,done,s_next,n_planning_updates):
        # TO DO: Add own code
        #make model
        self.counts[s, a, s_next] += 1
        self.rewardsum[s, a, s_next] += r
        self.estimate[ s, a, s_next] = self.counts[s, a, s_next]/np.sum(self.counts[s, a])
        self.rewestimate [s, a, s_next] = self.rewardsum[s, a, s_next]/self.counts[s, a, s_next]
        # fill priority queue
        a_next = np.random.choice(np.where(self.Q_sa[s_next, ] == np.amax(self.Q_sa[s_next, ]))[0])
        p = abs(r + self.gamma * self.Q_sa[s_next, a_next] - self.Q_sa[s,a])
        if (p > self.priority_cutoff):
            self.queue.put((-p, (s,a)))

        
        for i in range(n_planning_updates):
            if (self.queue.empty()):
                break
            pr = self.queue.get()
            state = pr[1][0]
            action = pr[1][1]

            est_state = np.random.choice(np.where(self.estimate[state, action] == np.amax(self.estimate[state, action]))[0])

            est_rew = self.rewestimate[state, action, est_state]
            a_next = np.random.choice(np.where(self.Q_sa[est_state, ] == np.amax(self.Q_sa[est_state, ]))[0])

            self.Q_sa[state, action] += self.learning_rate * ( est_rew + self.gamma * self.Q_sa[est_state, a_next] - self.Q_sa[state,action])
            #fill priority queue
            for s_ in range(self.n_states):
                for a_ in range(self.n_actions): 
                    if (self.counts[s_, a_, state] > 0):
                        est_rew = self.rewestimate[s_, a_, state]
                        p = abs(est_rew + self.gamma * self.Q_sa[state, action] - self.Q_sa[s_,a_])
                        if (p > self.priority_cutoff):
                            self.queue.put((-p, (s_,a_)))



class REINFORCE:
    def __init__(self, n_states, n_actions, learning_rate,  gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.dist = np.full((n_states, n_actions), 1/self.n_actions)

    def select_action(self, state):        
        return np.random.choice(self.n_actions, p = self.dist[state])


    def update(self, rewards, actions, states):
        
        tot_rewards = []
        for t in range(len(rewards)):
            tot_reward = 0
            for i in range(t):
                tot_reward += pow(self.gamma, i) * rewards[i]

            tot_rewards.append(tot_reward)

        for t in range(len(actions)):
            prob = self.dist[states[t],actions[t]]

            print("st, act: " + str(states[t]) + ", " + str(actions[t]) + "= " + str(prob))
            prob = np.log(prob)
            loss = float(-prob) * float(tot_rewards[t])* float(pow(self.gamma,t)) * float(self.learning_rate)
            last = self.dist[states[t],actions[t]]
            self.dist[states[t],actions[t]] += loss
            
            if self.dist[states[t],actions[t]] < 0.0001:
                self.dist[states[t],actions[t]] = 0.0001
                loss = self.dist[states[t],actions[t]] - last           
                
            elif  self.dist[states[t],actions[t]]  >1:
                self.dist[states[t],actions[t]]= 0.9999 
                loss = self.dist[states[t],actions[t]] - last           
           
            loss /= (len(self.dist[states[t] ])-1)

            for i in range(len(self.dist[states[t] ])):
                if i != actions[t]:
                    self.dist[states[t],actions[i]] -= loss  

        print(self.dist)
 
            


            # self.dist(states(t)) = np.gradient(self.dist(states(t)), loss)

            
def test():

    n_timesteps = 100
    gamma = 0.99

    # Algorithm parameters
    policy = 'grad' # 'ps' 
    epsilon = 0.1
    learning_rate = 0.5
    n_planning_updates = 5

    # Plotting parameters
    plot = True
    plot_optimal_policy = True
    step_pause = 0.0001
    
    # Initialize environment and policy
    env = WindyGridworld()
    if policy == 'dyna':
        pi = DynaAgent(env.n_states,env.n_actions,learning_rate,gamma) # Initialize Dyna policy
    elif policy == 'ps':    
        pi = PrioritizedSweepingAgent(env.n_states,env.n_actions,learning_rate,gamma) # Initialize PS policy
    
    elif policy == "grad":
        env = WindyGridworld()
        episodes = 100
        gamma = 1
        pi = REINFORCE(env.n_states, env.n_actions, learning_rate, gamma)

        s = env.reset()  
        continuous_mode = False
        
        for t in range(episodes):
            rewards = []
            actions = []
            states = []
            state = env.reset()

            done = False
            steps = 0
            while done == False and steps <= 100:
                action = pi.select_action(state)
                state, r, done = env.step(action)
                rewards.append(r)
                actions.append(action)
                states.append(state) 
                if plot:
                    env.render(Q_sa=pi.dist,plot_optimal_policy=plot_optimal_policy,
                       step_pause=step_pause)                   
                steps += 1
            pi.update(rewards, actions, states)
                
            
            # Ask user for manual or continuous execution
            if not continuous_mode:
                key_input = input("Press 'Enter' to execute next step, press 'c' to run full algorithm")
                continuous_mode = True if key_input == 'c' else False

            
    else:
        raise KeyError('Policy {} not implemented'.format(policy))
    
    # Prepare for running
    s = env.reset()  
    continuous_mode = False
            
    if (policy != "grad"):
    
        for t in range(n_timesteps):            
            # Select action, transition, update policy
            a = pi.select_action(s,epsilon)
            s_next,r,done = env.step(a)
            pi.update(s=s,a=a,r=r,done=done,s_next=s_next,n_planning_updates=n_planning_updates)
            
            # Render environment
            if plot:
                env.render(Q_sa=pi.Q_sa,plot_optimal_policy=plot_optimal_policy,
                        step_pause=step_pause)
                
            # Ask user for manual or continuous execution
            if not continuous_mode:
                key_input = input("Press 'Enter' to execute next step, press 'c' to run full algorithm")
                continuous_mode = True if key_input == 'c' else False

            # Reset environment when terminated
            if done:
                s = env.reset()
            else:
                s = s_next
                
    
if __name__ == '__main__':
    test()