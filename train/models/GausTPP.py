import torch


class GausTPP(torch.nn.Module):
    def __init__(self, device):
        super(GausTPP, self).__init__()
        self.mu = torch.rand(1, requires_grad=True, device=device)
        self.sigma = torch.rand(1, requires_grad=True, device=device)
        self.a = torch.rand(1, requires_grad=True, device=device)

    def forward(self, x, t):
        out = (1/(self.sigma*(2*3.14)**0.5)) * torch.exp((-(t-self.mu)**2/(2*self.sigma**2))) + self.a
        return torch.abs(out)

    def parameters(self):
        return iter((self.mu, self.sigma, self.a))
    
    def predict_sim(self, x, t):
        return self.forward(x, t)
