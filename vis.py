'''
24/06/2024 | Hari Prasad
An experiment with torch hooks for activation visualization.
'''
import pathlib
import torch
from torch.utils.tensorboard import SummaryWriter
from model import ModelNetwork

if __name__ == "__main__":
    # intialize summarywriter
    log_dir = pathlib.Path.cwd() / "tensorboard"
    writer = SummaryWriter(log_dir)


    # define a sample tensor
    input = torch.rand(64)
    # Add batch size
    input = input.unsqueeze(0)
    # Initalize the model
    model = ModelNetwork()


    def act_hook(layer, inp, out):
        '''
        layer (torch.nn.module): model layer to use
        inp (torch.tensor): input to the layer
        out (torch.tensor): output tot he layer
        '''
        print(repr(layer))
        writer.add_histogram(repr(layer), out)

    # add hook to layers
    model.fc1.register_forward_hook(act_hook)
    model.fc2.register_forward_hook(act_hook)
    model.fc3.register_forward_hook(act_hook)

    # predict using the model
    pred = model(input)

    print(pred)

