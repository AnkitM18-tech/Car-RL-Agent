#Import dependencies
import gym
from gym import version
import highway_env
import numpy as np
from stable_baselines import HER,SAC,PPO2,DQN

"""
#Roundabout Env
env = gym.make('roundabout-v0')
model = PPO2('MlpPolicy', env, verbose=1)

#Random action
# for i in range(10):
#     done = False
#     env.reset()
#     while not done:
#         env.render()
#         action = env.action_space.sample()
#         next_state, reward, done, info = env.step(action)
#         print(info)
#         print(done)

# env.close()

#Training and Visualizing Agent-1
model.learn(total_timesteps=1000)

#Save and Load model
model.save("roundabout")
model = PPO2.load("roundabout")

for i in range(10):
    done=False
    obs = env.reset()
    while not done:
        env.render()
        action, _states = model.predict(obs)
        next_state, reward, done, info = env.step(action)
        print(info)
        print(done)

env.close()
"""
"""
#Setting up AGent-2 Parking Gym ENV
env = gym.make("parking-v0")
#Random action
for i in range(10):
    done = False
    env.reset()
    while not done:
        env.render()
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        print(info)
        print(done)

env.close()

#Training our Agent-2
model = HER("MlpPolicy", env, SAC, n_sampled_goal=4, goal_selection_strategy="future", verbose=2)
model.learn(1000)

#Save Model
model.save("parkingagent")
model = HER.load("parkingagent", env=env)

#Visualizing Agent-2
for i in range(10):
    done=False
    obs = env.reset()
    while not done:
        env.render()
        action, _states = model.predict(obs)
        next_state, reward, done, info = env.step(action)
        print(info)
        print(done)

env.close()
"""
#Agent-3 Highway agent
env = gym.make("merge-v0")
#Random action
for i in range(10):
    done = False
    env.reset()
    while not done:
        env.render()
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        print(info)
        print(done)

env.close()

#Creating agent model and train
model = DQN("MlpPolicy", env, verbose=1)
model.learn(1000)
model2 = PPO2("MlpPolicy",env,verbose=1)
model2.learn(1000)

#Visualizing Agent-3
for i in range(10):
    done=False
    obs = env.reset()
    while not done:
        env.render()
        action, _states = model.predict(obs)
        next_state, reward, done, info = env.step(action)
        print(info)
        print(done)

env.close()

#Visualizing Agent-3
for i in range(10):
    done=False
    obs = env.reset()
    while not done:
        env.render()
        action, _states = model2.predict(obs)
        next_state, reward, done, info = env.step(action)
        print(info)
        print(done)

env.close()

#Save and Load model
model.save("highway")
model = DQN.load("highway", env=env)

model2.save("highway2")
model2 = DQN.load("highway2",env=env)