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

        self.flags = flags

        if flags.use_lorentz:

            # Create the constant for mapping the frequency w
            w_numpy = np.arange(flags.freq_low, flags.freq_high,
                                (flags.freq_high - flags.freq_low) / flags.num_spec_points)

            # Create the frequency tensor from numpy array, put variables on cuda if available
            cuda = True if torch.cuda.is_available() else False
            if cuda:
                self.w = torch.tensor(w_numpy).cuda()
            else:
                self.w = torch.tensor(w_numpy)

        input_size = 8
        fc_layer_size = flags.linear[-1]
        coupling_layer_size = self.flags.int_layer_size

        self.cyl1 = fc_NN(self.flags)
        self.cyl2 = fc_NN(self.flags)
        self.cyl3 = fc_NN(self.flags)
        self.cyl4 = fc_NN(self.flags)

        # self.int = int_Lor_param(self.flags)

        # self.int_w0 = int_Lor_param(self.flags)
        # self.int_wp = int_Lor_param(self.flags)
        # self.int_g = int_Lor_param(self.flags)

        self.cyl1_eps = cyl_eps(self.flags,fc_layer_size)
        self.cyl2_eps = cyl_eps(self.flags,fc_layer_size)
        self.cyl3_eps = cyl_eps(self.flags,fc_layer_size)
        self.cyl4_eps = cyl_eps(self.flags,fc_layer_size)
        self.lin_eps_inf = nn.Linear(input_size, 1, bias=True)
        torch.nn.init.uniform_(self.lin_eps_inf.weight, a=1, b=3)

    def forward(self, G):
        """
        The forward function which defines how the network is connected
        :param G: The input geometry (Since this is a forward network)
        :return: S: The 300 dimension spectra
        """
        out = G

        out1 = self.cyl1(G[:, 0::4])
        out2 = self.cyl2(G[:, 1::4])
        out3 = self.cyl3(G[:, 2::4])
        out4 = self.cyl4(G[:, 3::4])

        # int = self.int(G)
        # int_w0 = self.int_w0(G)
        # int_wp = self.int_wp(G)
        # int_g = self.int_g(G)
        # eps_inf = self.lin_eps_inf(F.relu(out))
        # d = self.d(F.relu(out))

        if self.flags.use_lorentz:

            eps_inf = F.relu(self.lin_eps_inf(G))
            eps_inf_out = eps_inf
            eps_inf = eps_inf.expand(out.size()[0], self.flags.num_spec_points).type(torch.cfloat)

            int = 0

            out1,w0,wp,g = self.cyl1_eps(out1,int)
            out2,x,y,z = self.cyl2_eps(out2,int)
            w0 = torch.cat((w0,x),dim=1)
            wp = torch.cat((wp, x), dim=1)
            g = torch.cat((g, x), dim=1)
            out3,x,y,z = self.cyl3_eps(out3,int)
            w0 = torch.cat((w0,x),dim=1)
            wp = torch.cat((wp, x), dim=1)
            g = torch.cat((g, x), dim=1)
            out4,x,y,z = self.cyl4_eps(out4,int)
            w0 = torch.cat((w0,x),dim=1)
            wp = torch.cat((wp, x), dim=1)
            g = torch.cat((g, x), dim=1)

            w0_out = w0
            wp_out = wp
            g_out = g

            # out1 = self.cyl1_eps(out1,int_w0,int_wp,int_g)
            # out2 = self.cyl1_eps(out2,int_w0,int_wp,int_g)
            # out3 = self.cyl1_eps(out3,int_w0,int_wp,int_g)
            # out4 = self.cyl1_eps(out4,int_w0,int_wp,int_g)

            w_2 = self.w.expand(G.size()[0], self.flags.num_spec_points)
            eps = (out1 + out2 + out3 + out4 + eps_inf).type(torch.cfloat)

            n = sqrt(eps)
            # self.test_var = n.imag.data.cpu().numpy()
            d, _ = torch.max(G[:, 4:], dim=1)
            d = d.unsqueeze(1).expand_as(n)
            # d = G[:,1].unsqueeze(1).expand_as(n)
            if self.flags.normalize_input:
                d = d * (self.flags.geoboundary[-1]-self.flags.geoboundary[-2]) * 0.5 + (self.flags.geoboundary[-1]+self.flags.geoboundary[-2]) * 0.5
            abs = torch.exp(-0.0033 * 4 * math.pi * mul(mul(d, n.imag),w_2))

            # R = div(square((n-ones).abs()),square((n+ones).abs()))
            # T_coeff = ones - R
            T = mul(div(4*n.real, add(square(n.real+1), square(n.imag))), abs).float()

            return T,w0_out,wp_out,g_out,eps_inf_out

        return out,_,_,_,eps_inf_out

class fc_NN(nn.Module):
    def __init__(self, flags):
        super(fc_NN, self).__init__()

        self.flags = flags
        self.linears = nn.ModuleList([])
        self.bn_linears = nn.ModuleList([])
        self.flags.linear[0] = 2
        for ind, fc_num in enumerate(self.flags.linear[0:-1]):  # Excluding the last one as we need intervals
            self.linears.append(nn.Linear(fc_num, self.flags.linear[ind + 1], bias=True))
            self.bn_linears.append(nn.BatchNorm1d(self.flags.linear[ind + 1], track_running_stats=True, affine=True))

    def forward(self, G):

        out = G
        for ind, (fc, bn) in enumerate(zip(self.linears, self.bn_linears)):
            # print(out.size())
            if ind < len(self.linears) - 0:
                out = F.relu(bn(fc(out)))  # ReLU + BN + Linear
            else:
                out = bn(fc(out))
        return out

class int_Lor_param(nn.Module):
    def __init__(self, flags):
        super(int_Lor_param, self).__init__()

        self.flags = flags
        input_size = 8
        coupling_layer_size = self.flags.int_layer_size

        self.input_coupl = nn.Linear(input_size,coupling_layer_size, bias=True)
        torch.nn.init.uniform_(self.input_coupl.weight, a=0, b=0.001)
        self.bn = nn.BatchNorm1d(coupling_layer_size, track_running_stats=True, affine=True)
        self.input_coupl0 = nn.Linear(coupling_layer_size,coupling_layer_size, bias=True)
        torch.nn.init.uniform_(self.input_coupl0.weight, a=0, b=0.001)
        self.bn0 = nn.BatchNorm1d(coupling_layer_size, track_running_stats=True, affine=True)
        self.input_coupl1 = nn.Linear(coupling_layer_size, self.flags.num_lorentz_osc, bias=True)
        torch.nn.init.uniform_(self.input_coupl1.weight, a=0, b=0.001)
        self.bn1 = nn.BatchNorm1d(self.flags.num_lorentz_osc, track_running_stats=True, affine=True)

    def forward(self, G):

        out = G
        out = F.relu(self.bn(self.input_coupl(out)))
        out = F.relu(self.bn0(self.input_coupl0(out)))
        out = self.bn1(self.input_coupl1(out))
        out = self.flags.int_layer_str*out
        return out

class cyl_eps(nn.Module):
    def __init__(self, flags, fc_layer_size):
        super(cyl_eps, self).__init__()

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

        # Last layer is the Lorentzian parameter layer
        self.lin_w0 = nn.Linear(fc_layer_size, self.flags.num_lorentz_osc, bias=False)
        torch.nn.init.uniform_(self.lin_w0.weight, a=0.0, b=0.02)
        self.lin_wp = nn.Linear(fc_layer_size, self.flags.num_lorentz_osc, bias=False)
        torch.nn.init.uniform_(self.lin_wp.weight, a=0.0, b=0.02)
        self.lin_g = nn.Linear(fc_layer_size, self.flags.num_lorentz_osc, bias=False)
        torch.nn.init.uniform_(self.lin_g.weight, a=0.0, b=0.02)

    # def forward(self, input, int_w0, int_wp, int_g):
    def forward(self, input, int):
        """
        The forward function which defines how the network is connected
        :param G: The input geometry (Since this is a forward network)
        :return: S: The 300 dimension spectra
        """
        out = input

        # If use lorentzian layer, pass this output to the lorentzian layer
        if self.use_lorentz:

            w0 = F.relu(self.lin_w0(F.relu(out)) + int)
            wp = F.relu(self.lin_wp(F.relu(out)) + int)
            g = F.relu(self.lin_g(F.relu(out)) + int)

            w0_out = w0
            wp_out = wp
            g_out = g

            w0 = w0.unsqueeze(2)
            wp = wp.unsqueeze(2)
            g = g.unsqueeze(2)

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
            j = torch.tensor([0 + 1j], dtype=torch.cfloat).expand_as(e2)
            # ones = torch.tensor([1+0j],dtype=torch.cfloat).expand_as(e2)
            if torch.cuda.is_available():
                j = j.cuda()
                # ones = ones.cuda()

            eps = add(e1, mul(e2, j)).type(torch.cfloat)

            return eps,w0_out,wp_out,g_out

        return out,out,out,out


