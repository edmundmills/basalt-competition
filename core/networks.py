def disable_gradients(network):
    for param in network.parameters():
        param.requires_grad = False
