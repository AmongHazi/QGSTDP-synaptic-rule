import torch
import torch.nn as nn
import numpy as np
import time
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
import os
from tqdm import tqdm
import random
import torch.optim as optim



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class channel_attention(nn.Module):
    def __init__(self, channel, ratio=16):
        super(channel_attention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, False),
            nn.ReLU(),
            nn.Linear(channel // ratio, channel, False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        max_pool_out = self.max_pool(x).view([b, c])
        avg_pool_out = self.avg_pool(x).view([b, c])

        max_fc_out = self.fc(max_pool_out)
        avg_fc_out = self.fc(avg_pool_out)
        out = max_fc_out + avg_fc_out
        out = self.sigmoid(out).view([b, c, 1, 1])
        return out * x

class spatial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(spatial_attention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, 1, padding, bias=False)
        self.sigmoid = nn.Sequential()

    def forward(self, x):
        b, c, h, w = x.size()
        print(x.size())
        max_pool_out, _ = torch.max(x, dim=1, keepdim=True)
        mean_pool_out = torch.mean(x, dim=1, keepdim=True)
        mean_pool_out = mean_pool_out
        print(np.shape(max_pool_out))
        print(np.shape(mean_pool_out))
        pool_out = torch.cat([max_pool_out, mean_pool_out], dim=1)
        out = self.conv(pool_out)
        out = self.sigmoid(out)
        return out * x


class RLAgent:
    """
    The relevant code for this part will be made public after the paper is accepted.
    """

    def __init__(self, num_inputs):
        self.num_inputs = num_inputs

    def select_action(self, pre, post):
        return 1  

    def update(self, pre, post, action, reward):
        pass  

class GSTDP_LIF_NeuronModel(nn.Module):
    def __init__(
        self,
        num_inputs,
        threshold,
        tau_membrane,
        tau_refract,
        alpha_plus,
        alpha_minus,
        initial_weights,
        tau_plus,
        tau_minus
    ):
        super().__init__()

        self.num_inputs = num_inputs
        self.threshold = threshold
        self.tau_membrane = tau_membrane
        self.tau_refract = tau_refract
        self.alpha_plus = alpha_plus
        self.alpha_minus = alpha_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus

        self.weights = nn.Parameter(
            torch.randn(num_inputs, num_inputs) * initial_weights,
            requires_grad=False
        )

        self.membrane_potential = torch.zeros(num_inputs)
        self.refractory_period = torch.zeros(num_inputs)

        self.rl_agent = RLAgent(num_inputs)

    def fire_neurons(self, input_signal):
        self.membrane_potential += input_signal
        self.refractory_period = torch.clamp(self.refractory_period - 1, min=0)

        spike_mask = (self.membrane_potential >= self.threshold) & (self.refractory_period == 0)
        spikes = spike_mask.float()

        self.membrane_potential = torch.where(
            spike_mask,
            torch.zeros_like(self.membrane_potential),
            self.membrane_potential
        )
        self.refractory_period += spike_mask * self.tau_refract
        return spikes

    def update_weights(self, spike_times):
        with torch.no_grad():
            spike_times = spike_times.to(self.weights.device)

            for idx in range(len(spike_times)):
                pre = spike_times[idx].long().item()
                if pre >= self.num_inputs:
                    continue

                for j in range(idx + 1, len(spike_times)):
                    post = spike_times[j].long().item()
                    if post >= self.num_inputs:
                        continue

                    delta_t = spike_times[j] - spike_times[idx]
                    delta_t = delta_t.float()

                    if delta_t > 0:
                        stdp_ltp = self.alpha_plus * torch.exp(
                            -(delta_t ** 2) / (2 * self.tau_plus ** 2)
                        )

                        action = self.rl_agent.select_action(pre, post)

                        if action == 1:
                            self.weights[pre, post] += stdp_ltp
                            reward = stdp_ltp.item()
                        else:
                            reward = 0.0

                        self.rl_agent.update(pre, post, action, reward)

                    else:
                        stdp_ltd = -self.alpha_minus * torch.exp(
                            -(delta_t ** 2) / (2 * self.tau_minus ** 2)
                        )
                        self.weights[pre, post] += stdp_ltd

            self.weights.clamp_(0.0, 1.0)

    def forward(self, input_spikes):
        input_signal = input_spikes.float() * self.threshold
        spikes = self.fire_neurons(input_signal)

        spike_times = torch.nonzero(spikes, as_tuple=False).squeeze()
        if spike_times.ndim == 0:
            spike_times = spike_times.unsqueeze(0)

        if spike_times.numel() > 1:
            self.update_weights(spike_times)

        return spikes
