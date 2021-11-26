import torch


class PoissonTPP(torch.nn.Module):
    def __init__(self, device):
        super(PoissonTPP, self).__init__()
        self.a = torch.randn(1, 1, requires_grad=True, device=device)
        self.b = torch.randn(1, requires_grad=True, device=device)

    def forward(self, x, t):
        out = torch.abs(self.b)
        return out

    def parameters(self):
        return iter((self.a, self.b))
    
    def predict_sim(self, x, t):
        return self.forward(x, t)