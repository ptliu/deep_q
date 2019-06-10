import gym
import agent
import torch
import torchvision.transforms as transforms
from PIL import Image
from collections import namedtuple
import numpy as np

def get_cart_location(screen_width, env):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  

def get_screen(env, device):
    resize = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize(40, interpolation=Image.CUBIC),
                                 transforms.ToTensor()])


    screen = env.render(mode='rgb_array')
    screen = screen.transpose((2, 0, 1))
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width, env)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    screen = screen[:, :, slice_range]
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    return resize(screen).unsqueeze(0).to(device)



if __name__ == "__main__":
    env = gym.make('CartPole-v0').unwrapped
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    env.reset()
    screen = get_screen(env, device)
    _, _, width, height = screen.shape
    obs = env.reset()
    n_actions = env.action_space.n
    agent = agent.Agent(width, height, n_actions, env)
    agent.train()
    """
    for _ in range(1000):
        env.render()
        obs, reward, done, info = env.step(env.action_space.sample())

        print(obs)
        if done:
            print("episode done")
            break
        
    """
    env.close()

