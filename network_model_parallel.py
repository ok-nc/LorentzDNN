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
        # self.cyl2 = fc_NN(self.flags)
        # self.cyl3 = fc_NN(self.flags)
        # self.cyl4 = fc_NN(self.flags)

        self.input_coupl0 = nn.Linear(input_size,coupling_layer_size, bias=False)
        self.bn0 = nn.BatchNorm1d(coupling_layer_size, track_running_stats=True, affine=True)
        self.input_coupl1 = nn.Linear(coupling_layer_size,coupling_layer_size, bias=False)
        self.bn1 = nn.BatchNorm1d(coupling_layer_size, track_running_stats=True, affine=True)
        self.input_coupl2 = nn.Linear(coupling_layer_size, self.flags.num_lorentz_osc, bias=False)
        self.bn2 = nn.BatchNorm1d(self.flags.num_lorentz_osc, track_running_stats=True, affine=True)
        # self.coupling_layer = nn.Linear(fc_layer_size,coupling_layer_size)
        # self.bn_coupling = nn.BatchNorm1d(coupling_layer_size, track_running_stats=True, affine=True)
        # self.trans = Trans_calc_NN(self.flags,coupling_layer_size)

        self.cyl1_eps = cyl_eps(self.flags,fc_layer_size)
        self.cyl2_eps = cyl_eps(self.flags,fc_layer_size)
        self.cyl3_eps = cyl_eps(self.flags,fc_layer_size)
        self.cyl4_eps = cyl_eps(self.flags,fc_layer_size)
        self.lin_eps_inf = nn.Linear(input_size, 1, bias=True)

    def forward(self, G):
        """
        The forward function which defines how the network is connected
        :param G: The input geometry (Since this is a forward network)
        :return: S: The 300 dimension spectra
        """
        out = G

        out1 = self.cyl1(G[:, 0::4])
        out2 = self.cyl1(G[:, 1::4])
        out3 = self.cyl1(G[:, 2::4])
        out4 = self.cyl1(G[:, 3::4])
        int = self.bn0(self.input_coupl0(G))
        int = self.bn1(self.input_coupl1(F.relu(int)))
        int2 = self.bn2(self.input_coupl2(F.relu(int)))
        # eps_inf = self.lin_eps_inf(F.relu(out))
        # d = self.d(F.relu(out))

        if self.flags.use_lorentz:

            eps_inf = self.lin_eps_inf(F.relu(G))
            eps_inf = eps_inf.expand(out.size()[0], self.flags.num_spec_points).type(torch.cfloat)

            out1 = self.cyl1_eps(out1, self.flags.int_layer_str*int2)
            out2 = self.cyl2_eps(out2, self.flags.int_layer_str*int2)
            out3 = self.cyl3_eps(out3, self.flags.int_layer_str*int2)
            out4 = self.cyl4_eps(out4, self.flags.int_layer_str*int2)

            eps = (out1 + out2 + out3 + out4 + eps_inf).type(torch.cfloat)

            n = sqrt(eps)
            # self.test_var = n.imag.data.cpu().numpy()
            d, _ = torch.max(G[:, 4:], dim=1)
            d = d.unsqueeze(1).expand_as(n)
            # d = G[:,1].unsqueeze(1).expand_as(n)
            if self.flags.normalize_input:
                d = d * (self.flags.geoboundary[-1]-self.flags.geoboundary[-2]) * 0.5 + (self.flags.geoboundary[-1]+self.flags.geoboundary[-2]) * 0.5
            alpha = torch.exp(-0.0005 * 4 * math.pi * mul(d, n.imag))

            # R = div(square((n-ones).abs()),square((n+ones).abs()))
            # T_coeff = ones - R
            T = mul(div(4*n.real, add(square(n.real+1), square(n.imag))), alpha).float()

            return T,_,_,_

        return out,_,_,_


class lorNN(nn.Module):
    def __init__(self, flags):
        super(lorNN, self).__init__()

        # Set up whether this uses a Lorentzian oscillator, this is a boolean value
        self.use_lorentz = flags.use_lorentz
        self.flags = flags
        self.flags.linear[0] = 2
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
        for ind, fc_num in enumerate(flags.linear[0:-1]):  # Excluding the last one as we need intervals
            self.linears.append(nn.Linear(fc_num, flags.linear[ind + 1], bias=True))
            # torch.nn.init.uniform_(self.linears[ind].weight, a=1, b=2)

            self.bn_linears.append(nn.BatchNorm1d(flags.linear[ind + 1], track_running_stats=True, affine=True))

        layer_size = flags.linear[-1]

        # Last layer is the Lorentzian parameter layer
        self.lin_w0 = nn.Linear(layer_size, self.flags.num_lorentz_osc, bias=False)
        self.lin_wp = nn.Linear(layer_size, self.flags.num_lorentz_osc, bias=False)
        self.lin_g = nn.Linear(layer_size, self.flags.num_lorentz_osc, bias=False)
        # self.lin_eps_inf = nn.Linear(layer_size, 1, bias=True)

    def forward(self, G):
        """
        The forward function which defines how the network is connected
        :param G: The input geometry (Since this is a forward network)
        :return: S: The 300 dimension spectra
        """
        out = G

        # For the linear part
        for ind, (fc, bn) in enumerate(zip(self.linears, self.bn_linears)):
            # print(out.size())
            if ind < len(self.linears) - 0:
                out = F.relu(bn(fc(out)))  # ReLU + BN + Linear
            else:
                out = bn(fc(out))

        # If use lorentzian layer, pass this output to the lorentzian layer
        if self.use_lorentz:

            w0 = F.relu(self.lin_w0(F.relu(out)))
            wp = F.relu(self.lin_wp(F.relu(out)))
            g = F.relu(self.lin_g(F.relu(out)))
            # eps_inf = self.lin_eps_inf(F.relu(out))

            # w0_out = w0
            # wp_out = wp
            # g_out = g

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
            # eps_inf = eps_inf.expand_as(e1).type(torch.cfloat)
            # e1 += eps_inf
            j = torch.tensor([0 + 1j], dtype=torch.cfloat).expand_as(e2)
            # ones = torch.tensor([1+0j],dtype=torch.cfloat).expand_as(e2)
            if torch.cuda.is_available():
                j = j.cuda()
                # ones = ones.cuda()

            eps = add(e1, mul(e2, j))
            return eps

            # n = sqrt(eps)
            # self.test_var = n.imag.data.cpu().numpy()
            # d, _ = torch.max(G[:, 4:], dim=1)
            # d = d.unsqueeze(1).expand_as(n)
            # # d = G[:,1].unsqueeze(1).expand_as(n)
            # if self.flags.normalize_input:
            #     d = d * (self.flags.geoboundary[-1]-self.flags.geoboundary[-2]) * 0.5 + (self.flags.geoboundary[-1]+self.flags.geoboundary[-2]) * 0.5
            # alpha = torch.exp(-0.0005 * 4 * math.pi * mul(d, n.imag))
            #
            # # R = div(square((n-ones).abs()),square((n+ones).abs()))
            # # T_coeff = ones - R
            # T = mul(div(4*n.real, add(square(n.real+1), square(n.imag))), alpha).float()

            # return T

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
        torch.nn.init.uniform_(self.lin_w0.weight, a=0.0, b=0.1)
        self.lin_wp = nn.Linear(fc_layer_size, self.flags.num_lorentz_osc, bias=False)
        torch.nn.init.uniform_(self.lin_wp.weight, a=0.0, b=0.1)
        self.lin_g = nn.Linear(fc_layer_size, self.flags.num_lorentz_osc, bias=False)
        torch.nn.init.uniform_(self.lin_g.weight, a=0.0, b=0.05)

    def forward(self, input, in_coupl_out):
        """
        The forward function which defines how the network is connected
        :param G: The input geometry (Since this is a forward network)
        :return: S: The 300 dimension spectra
        """
        out = input

        # If use lorentzian layer, pass this output to the lorentzian layer
        if self.use_lorentz:

            w0 = F.relu(self.lin_w0(F.relu(out)) + in_coupl_out)
            wp = F.relu(self.lin_wp(F.relu(out)) + in_coupl_out)
            g = F.relu(self.lin_g(F.relu(out)) + in_coupl_out)

            # w0_out = w0
            # wp_out = wp
            # g_out = g

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
            j = torch.tensor([0 + 1j], dtype=torch.cfloat).expand_as(e2)
            # ones = torch.tensor([1+0j],dtype=torch.cfloat).expand_as(e2)
            if torch.cuda.is_available():
                j = j.cuda()
                # ones = ones.cuda()

            eps = add(e1, mul(e2, j)).type(torch.cfloat)

            return eps

        return out


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


class Trans_calc_NN(nn.Module):
    def __init__(self, flags, coupling_layer_size):
        super(Trans_calc_NN, self).__init__()

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


        layer_size = flags.linear[-1]

        # Last layer is the Lorentzian parameter layer
        self.lin_w0 = nn.Linear(coupling_layer_size, self.flags.num_lorentz_osc, bias=False)
        self.lin_wp = nn.Linear(coupling_layer_size, self.flags.num_lorentz_osc, bias=False)
        self.lin_g = nn.Linear(coupling_layer_size, self.flags.num_lorentz_osc, bias=False)
        self.lin_eps_inf = nn.Linear(coupling_layer_size, 1, bias=True)

    def forward(self, input, G):
        """
        The forward function which defines how the network is connected
        :param G: The input geometry (Since this is a forward network)
        :return: S: The 300 dimension spectra
        """
        out = input

        # If use lorentzian layer, pass this output to the lorentzian layer
        if self.use_lorentz:

            w0 = F.relu(self.lin_w0(F.relu(out)))
            wp = F.relu(self.lin_wp(F.relu(out)))
            g = F.relu(self.lin_g(F.relu(out)))
            eps_inf = self.lin_eps_inf(F.relu(out))

            # w0_out = w0
            # wp_out = wp
            # g_out = g

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
            j = torch.tensor([0 + 1j], dtype=torch.cfloat).expand_as(e2)
            # ones = torch.tensor([1+0j],dtype=torch.cfloat).expand_as(e2)
            if torch.cuda.is_available():
                j = j.cuda()
                # ones = ones.cuda()

            eps = add(e1, mul(e2, j))

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

            return T

        return out