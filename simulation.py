import gym

env = gym.make('CartPole-v0')
obs = env.reset()
print(env.action_space)
for act in env.action_space:
    print("action: ", act)
for _ in range(1000):
    env.render()
    obs, reward, done, info = env.step(env.action_space.sample())

    print(obs)
    if done:
        print("episode done")
        break
    

env.close()

