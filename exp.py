import numpy as np
from env import WindyGridworld
from agents import DynaAgent,PrioritizedSweepingAgent, REINFORCE
from help import LearningCurvePlot, smooth

def run_repetitions(policy, n_repetitions, n_timesteps, smoothing_window, learning_rate, gamma,
                    epsilon, n_planning_updates):
    # Write all your experiment code here
    # Look closely at the code in test() of MBRLAgents.py for an example of the execution loop
    # Log the obtained rewards during a single training run of n_timesteps, and repeat this proces n_repetitions times
    # Average the learning curves over repetitions, and then additionally smooth the curve
    # Be sure to turn environment rendering off! It heavily slows down your runtime
    env = WindyGridworld()
    if policy == 'Dyna':
        pi = DynaAgent(env.n_states,env.n_actions,learning_rate,gamma) # Initialize Dyna policy
    elif policy == 'Prioritized Sweeping':    
        pi = PrioritizedSweepingAgent(env.n_states,env.n_actions,learning_rate,gamma) # Initialize PS policy
    elif policy == 'REINFORCE':
        pi = REINFORCE(env.n_states,env.n_actions,learning_rate,gamma)
    else:
        raise KeyError('Policy {} not implemented'.format(policy))

    if policy == 'Dyna' or policy == 'Prioritized Sweeping':
        tot_reward = np.zeros(n_timesteps)
        for i in range(n_repetitions):
            s = env.reset()  
            for t in range(n_timesteps):                        
                a = pi.select_action(s,epsilon)
                s_next,r,done = env.step(a)
                pi.update(s=s,a=a,r=r,done=done,s_next=s_next,n_planning_updates=n_planning_updates)
                
                if done:
                    s = env.reset()
                else:
                    s = s_next                
                tot_reward[t] += r                 

        for k in range(n_timesteps):
            tot_reward[k] /= n_repetitions
            
        
    else:
        tot_reward = np.zeros(n_timesteps)
        for i in range(n_repetitions):
            s = env.reset()            
            for t in range(n_timesteps):
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
                    steps += 1
                pi.update(rewards, actions, states)
                tot_reward[t] += np.sum(rewards)
        for k in range(n_timesteps):
                tot_reward[k] /= n_repetitions

    learning_curve = tot_reward[:]
    # Apply additional smoothing
    learning_curve = smooth(learning_curve, smoothing_window) # additional smoothing
    return learning_curve



def experiment():

    n_timesteps = 1000
    n_repetitions = 100
    smoothing_window = 101
    gamma = 0.99

    learning_rate = 0.5
    n_planning_updates = 5
    epsilon = 0.5
    policy = 'REINFORCE'
    Plot = LearningCurvePlot(title = 'Comparing Dyna, Prioritized Sweeping and REINFORCE')        
    learning_curve=run_repetitions(policy, n_repetitions, n_timesteps, smoothing_window, 
                                        learning_rate, gamma, epsilon, n_planning_updates)
    Plot.add_curve(learning_curve,label='$\policy$ = {}'.format(policy))  
    Plot.save('Reinforce.png')

    # for policy in ['Dyna','Prioritized Sweeping', 'REINFORCE']:
    #     learning_rate = 0.5
    #     n_planning_updates = 5
    #     epsilon = 0.5
    #     Plot = LearningCurvePlot(title = 'Comparing Dyna, Prioritized Sweeping and REINFORCE')        
    #     learning_curve=run_repetitions(policy, n_repetitions, n_timesteps, smoothing_window, 
    #                                        learning_rate, gamma, epsilon, n_planning_updates)
    #     Plot.add_curve(learning_curve,label='$\policy$ = {}'.format(policy))  
    # Plot.save('compare.png')

    
    
if __name__ == '__main__':
    experiment()