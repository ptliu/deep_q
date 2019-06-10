
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as torch_init
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torchvision.transforms as transforms
from PIL import Image
from collections import namedtuple
from itertools import count
import math

def get_cart_location(screen_width, env):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

def get_screen(env, device):
    resize = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize(40, interpolation=Image.CUBIC),
                                 transforms.ToTensor()])


    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW)

    screen = env.render(mode='rgb_array').transpose((2, 0, 1))

    # Cart is in the lower half, so strip off the top and bottom of the screen
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
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)

def plot_durations(episode_durations):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    #plt.show()



class ReplayMemory():

    def __init__(self, capacity):

        self.Transition = namedtuple('Transition',
                               ('state', 'action', 'next_state', 'reward'))
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


class Agent():
    def __init__(self, width, height, outputs, env):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(width=width, height=height, outputs=outputs).to(self.device)
        self.target_net = DQN(width, height, outputs).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(10000)
        self.steps = 0
        self.outputs = outputs
        self.width = width
        self.height = height
        self.update_frequency = 128
        self.batch_size = 128
        self.env = env
        self.episode_durations = []
        self.update_freq = 10
        self.Transition = namedtuple('Transition',
                               ('state', 'action', 'next_state', 'reward'))


    def select_action(self,state):
        self.steps += 1
 #       if(random.random() < (0.05 + (0.99 - 0.05) * math.exp(-1.0 * self.steps/ 200))):
  #          return torch.tensor([[random.randrange(self.outputs)]], device=self.device, dtype=torch.long)
   #     else:
            #exploit
    #        with torch.no_grad():
        return self.policy_net(state).max(1)[1].view(1,1)

    def step(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = self.Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=self.device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * 0.999) + reward_batch

        loss = func.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
       # for param in self.policy_net.parameters():
            #param.grad.data.clamp_(-1, 1)
        self.optimizer.step() 


    def train(self, num_episodes=1000):
        for i_episode in range(num_episodes):
            # Initialize the environment and state
            self.env.reset()
            last_screen = get_screen(self.env, self.device)
            current_screen = get_screen(self.env, self.device)
            state = current_screen - last_screen
            for t in count():
                # Select and perform an action
                action = self.select_action(state)
                _, reward, done, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=self.device)

                # Observe new state
                last_screen = current_screen
                current_screen = get_screen(self.env, self.device)
                if not done:
                    next_state = current_screen - last_screen
                else:
                    next_state = None

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                self.step()
                if done:
                    self.episode_durations.append(t + 1)
                    plot_durations(self.episode_durations)
                    break
            # Update the target network, copying all weights and biases in DQN
            if i_episode % self.update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        plt.show()
    
    def save(target_filename='target.pkl', policy_filename='policy.pkl'):
        import pickle
        with open(target_filename, 'wb') as f:
            torch.save(self.target_net, f)
        with open(policy_filename, 'wb') as f:
            torch.save(self.target_net, f)
    def load(target_filename='target.pkl', policy_filename='policy.pkl'):
        import pickle
        with open(target_filename, 'rb') as f:
            self.target_net = torch.load(f)
        with open(policy_filename, 'rb') as f:
            self.policy_net = torch.load(f)



class DQN(nn.Module):
    """ A basic convolutional neural network model for baseline comparison. 
    
    """
    
    def __init__(self, width, height, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2)#cuda()
        
        # Add batch-normalization to the outputs of conv1
        self.conv1_normed = nn.BatchNorm2d(16)
        
        # Initialized weights using the Xavier-Normal method
        torch_init.xavier_normal_(self.conv1.weight)


        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2)#.cuda()
        self.conv2_normed = nn.BatchNorm2d(32)
        torch_init.xavier_normal_(self.conv2.weight)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2)#.cuda()
        self.conv3_normed = nn.BatchNorm2d(32)
        torch_init.xavier_normal_(self.conv3.weight)


        conv_width = self.conv2d_size_out( self.conv2d_size_out( self.conv2d_size_out(width)))

        conv_height = self.conv2d_size_out( self.conv2d_size_out( self.conv2d_size_out(height)))
        self.fc1 = nn.Linear(in_features=conv_width * conv_height * 32, out_features=outputs)#.cuda()
        self.fc1_normed = nn.BatchNorm1d(outputs)
        torch_init.xavier_normal_(self.fc1.weight)

        #self.fc2 = nn.Linear(in_features=128, out_features=outputs).cuda()
        #torch_init.xavier_normal_(self.fc2.weight)

    def conv2d_size_out(self,size, kernel_size=5, stride=2): 
        return (size - (kernel_size - 1) - 1) // stride + 1

    def forward(self, batch):
        
        # Apply first convolution, followed by ReLU non-linearity; 
        # use batch-normalization on its outputs
        batch = func.relu(self.conv1_normed(self.conv1(batch)))
        
        # Apply conv2 and conv3 similarly
        batch = func.relu(self.conv2_normed(self.conv2(batch)))
        batch = func.relu(self.conv3_normed(self.conv3(batch)))
        
        # Pass the output of conv3 to the pooling layer
        #batch = self.pool(batch)

        # Reshape the output of the conv3 to pass to fully-connected layer
        batch = batch.view(-1, self.num_flat_features(batch))
        
        # Connect the reshaped features of the pooled conv3 to fc1
        # Using activation function of: relu here - is this necessary? 
        batch = self.fc1(batch)#func.relu(self.fc1_normed(self.fc1(batch)))
        
        # Connect fc1 to fc2 - this layer is slightly different than the rest (why?)
        # A fully connected layer to another fully connected layer 
        #batch = self.fc2(batch)


        # Return the class predictions
        #TODO: apply an activition function to 'batch'
        # Applying sigmoid on 'batch' (output of fc2)
        return batch#func.sigmoid(batch)
    
    

    def num_flat_features(self, inputs):
        
        # Get the dimensions of the layers excluding the inputs
        size = inputs.size()[1:]
        # Track the number of features
        num_features = 1
        
        for s in size:
            num_features *= s
        
        return num_features
  
