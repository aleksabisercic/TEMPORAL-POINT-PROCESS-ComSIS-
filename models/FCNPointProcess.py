import torch
from torch import nn


class FCNPointProcess(nn.Module):
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def __init__(self, input_size, output_size, dropout=0, num_layers=2, step=2):
        super(FCNPointProcess, self).__init__()

        lst = nn.ModuleList()
        # Fully connected layer
        out_size = input_size
        for i in range(num_layers):
            inp_size = out_size
            out_size = int(inp_size//step)
            if i == num_layers-1:
                block = nn.Linear(inp_size, 1)
            else:
                block = nn.Sequential(nn.Linear(inp_size, out_size),
                                      nn.ReLU())
                block.apply(FCNPointProcess.init_weights)
            lst.append(block)

        self.fc = nn.Sequential(*lst)

    def forward(self, x, t):
        interevents = torch.abs(t-x)
        t = interevents.to(torch.float32)
        # x = torch.cat((x, t.reshape(1, -1)), dim=1)
        x = t.reshape(1, -1)
        nn_out = self.fc(x)
        out = torch.abs(nn_out)
        return out
