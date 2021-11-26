import torch


class PoissonPolynomialTPP(torch.nn.Module):
    def __init__(self, device):
        super(PoissonPolynomialTPP, self).__init__()
        self.a = torch.randn(1, requires_grad=True, device=device)
        self.b = torch.randn(1, requires_grad=True, device=device)
        self.c = torch.randn(1, requires_grad=True, device=device)

    def forward(self, x, t):
        out = self.a + self.b*t + self.c*t**2
        return torch.abs(out)

    def parameters(self):
        return iter((self.a, self.b, self.c))
    
    def predict_sim(self, x, t):
        return self.forward(x, t)
