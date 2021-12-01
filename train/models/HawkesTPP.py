import torch


class HawkesTPP(torch.nn.Module):
    def __init__(self, device):
        super(HawkesTPP, self).__init__()
        self.alpha = torch.tensor(0.1842, requires_grad=True, device=device)  
        self.mu = torch.tensor(1.7657, requires_grad=True, device=device)
        # self.mu = torch.nn.parameter.Parameter(torch.ones(1)*1.608, requires_grad=True)
        # self.alpha = torch.nn.parameter.Parameter(torch.ones(1)*0.0, requires_grad=True)
        # self.mu = torch.nn.parameter.Parameter(torch.ones(1)*0.9, requires_grad=True)
        # self.alpha = torch.nn.parameter.Parameter(torch.ones(1)*0.5, requires_grad=True)

        # self.alpha = torch.rand(1, requires_grad=True, device=device)
        # self.mu = torch.rand(1, requires_grad=True, device=device)
        # self.par = torch.rand(1, requires_grad=True, device=device)
        
        # self.power = torch.zeros(1, requires_grad=False)

    def forward(self, x, t):
        interevents = torch.abs(t-x)  #nisam siguran da je ovako definisano u literaturi, mada je slično
        exp_interevents = torch.exp((-interevents)) # uvek vraca 0 jer je e-(__) pribilizno 0
        variable_part = torch.sum(exp_interevents).reshape(1, -1)
        # print(variable_part)
        out = torch.abs(self.mu) + torch.abs(self.alpha) * variable_part
        return out

    def parameters(self):
        return iter((self.mu, self.alpha))

    def predict_sim(self, x, t):
        return self.forward(x, t)
