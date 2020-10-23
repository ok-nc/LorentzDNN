"""
This is the module where the model is defined. It uses the nn.Module as a backbone to create the network structure
"""

import math
import numpy as np

# Pytorch module
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import pow, add, mul, div, sqrt, square

class Forward(nn.Module):
    def __init__(self, flags):
        super(Forward, self).__init__()

        # Set up whether this uses a Lorentzian oscillator, this is a boolean value
        self.use_lorentz = flags.use_lorentz
        self.flags = flags

        if flags.use_lorentz:

            # Create the constant for mapping the frequency w
            w_numpy = np.arange(flags.freq_low, flags.freq_high,
                                (flags.freq_high - flags.freq_low) / self.flags.num_spec_points)

            # Create eps_inf variable, currently set to a constant value
            # self.epsilon_inf = torch.tensor([5+0j],dtype=torch.cfloat)

            # Create the frequency tensor from numpy array, put variables on cuda if available
            cuda = True if torch.cuda.is_available() else False
            if cuda:
                self.w = torch.tensor(w_numpy).cuda()
                # self.epsilon_inf = self.epsilon_inf.cuda()
            else:
                self.w = torch.tensor(w_numpy)

        """
        General layer definitions:
        """
        # Linear Layer and Batch_norm Layer definitions here
        self.linears = nn.ModuleList([])
        self.bn_linears = nn.ModuleList([])
        for ind, fc_num in enumerate(flags.linear[0:-1]):               # Excluding the last one as we need intervals
            self.linears.append(nn.Linear(fc_num, flags.linear[ind + 1], bias=True))
            # torch.nn.init.uniform_(self.linears[ind].weight, a=1, b=2)

            self.bn_linears.append(nn.BatchNorm1d(flags.linear[ind + 1], track_running_stats=True, affine=True))

        layer_size = flags.linear[-1]

        # Last layer is the Lorentzian parameter layer
        self.lin_w0 = nn.Linear(layer_size, self.flags.num_lorentz_osc, bias=False)
        self.lin_wp = nn.Linear(layer_size, self.flags.num_lorentz_osc, bias=False)
        self.lin_g = nn.Linear(layer_size, self.flags.num_lorentz_osc, bias=False)
        self.lin_eps_inf = nn.Linear(layer_size, 1, bias=True)

    def forward(self, G):
        """
        The forward function which defines how the network is connected
        :param G: The input geometry (Since this is a forward network)
        :return: S: The 300 dimension spectra
        """
        out = G

        # For the linear part
        for ind, (fc, bn) in enumerate(zip(self.linears, self.bn_linears)):
            #print(out.size())
            if ind < len(self.linears) - 0:
                out = F.relu(bn(fc(out)))                                   # ReLU + BN + Linear
            else:
                out = bn(fc(out))

        # If use lorentzian layer, pass this output to the lorentzian layer
        if self.use_lorentz:

            w0 = F.relu(self.lin_w0(F.relu(out)))
            wp = F.relu(self.lin_wp(F.relu(out)))
            g = F.relu(self.lin_g(F.relu(out)))
            eps_inf = self.lin_eps_inf(F.relu(out))


            w0_out = w0
            wp_out = wp
            g_out = g

            w0 = w0.unsqueeze(2) * 1
            wp = wp.unsqueeze(2) * 1
            g = g.unsqueeze(2) * 0.1


             # Expand them to parallelize, (batch_size, #osc, #spec_point)
            wp = wp.expand(out.size()[0], self.flags.num_lorentz_osc, self.flags.num_spec_points)
            w0 = w0.expand_as(wp)
            g = g.expand_as(w0)
            w_expand = self.w.expand_as(g)


            # Define dielectric function (real and imaginary parts separately)
            num1 = mul(square(wp), add(square(w0), -square(w_expand)))
            num2 = mul(square(wp), mul(w_expand, g))
            denom = add(square(add(square(w0), -square(w_expand))), mul(square(w_expand), square(g)))
            e1 = div(num1, denom)
            e2 = div(num2, denom)

            # self.e2 = e2.data.cpu().numpy()                 # This is for plotting the imaginary part
            # # self.e1 = e1.data.cpu().numpy()                 # This is for plotting the imaginary part

            e1 = torch.sum(e1, 1).type(torch.cfloat)
            e2 = torch.sum(e2, 1).type(torch.cfloat)
            eps_inf = eps_inf.expand_as(e1).type(torch.cfloat)
            e1 += eps_inf
            j = torch.tensor([0+1j],dtype=torch.cfloat).expand_as(e2)
            # ones = torch.tensor([1+0j],dtype=torch.cfloat).expand_as(e2)
            if torch.cuda.is_available():
                j = j.cuda()
                # ones = ones.cuda()

            eps = add(e1, mul(e2,j))
            n = sqrt(eps)
            self.test_var = n.imag.data.cpu().numpy()
            d, _ = torch.max(G[:, 4:], dim=1)
            d = d.unsqueeze(1).expand_as(n)
            # d = G[:,1].unsqueeze(1).expand_as(n)
            if self.flags.normalize_input:
                d = d * (self.flags.geoboundary[-1]-self.flags.geoboundary[-2]) * 0.5 + (self.flags.geoboundary[-1]+self.flags.geoboundary[-2]) * 0.5
            alpha = torch.exp(-0.0005 * 4 * math.pi * mul(d, n.imag))

            # R = div(square((n-ones).abs()),square((n+ones).abs()))
            # T_coeff = ones - R
            T = mul(div(4*n.real, add(square(n.real+1), square(n.imag))), alpha).float()

            return T, w0_out, wp_out, g_out

        return out,out,out,out
