from torch import nn

def disable_gradients(network: nn.Module):
    """Freezes the parameters in the input network."""
    for param in network.parameters():
        param.requires_grad = False
