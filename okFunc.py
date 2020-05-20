import torch
from torch import div

# Custom function to manipulate gradients in-place

class scale_grad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # print(torch.max(grad_input))
        # grad_input = grad_input * 1.00
        grad_input = torch.clamp(grad_input, min=-0.5, max=0.5)
        print(torch.max(grad_input))
        return grad_input