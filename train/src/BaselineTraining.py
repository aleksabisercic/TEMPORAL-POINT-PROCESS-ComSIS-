import math

import matplotlib.pyplot as plt
import torch
from scipy.special import p_roots
import numpy as np
import pickle


def integral(model, time, in_size, no_steps, device, h=None, atribute=None, method="Trapezoid", farest_interevent=-1e9):

    def integral_solve(z0, time_to_t0, t0, t1, atribute, no_steps=10, h=None, method="Trapezoid"):
        if no_steps is not None:
            h_max = (t1 - t0)/no_steps  #h - lenght of segment
        elif h is not None:
            no_steps = math.ceil((t1 - t0)/h) #ako je zadat h(duzina intervala)
            h_max = (t1 - t0)/no_steps

        integral = 0
        t = t0
        
        def Gaussian_quadrature(n, lower_l, upper_l):
            m = (upper_l-lower_l)/2
            c = (upper_l+lower_l)/2
            [x,w] = p_roots(n+1)
            weights = m*w
            time_train_integral = m*x+c
            return time_train_integral, weights

        if method == "Euler":
            for _ in range(no_steps):
                integral += z0*h_max
                t = t + h_max
                # atribute = atribute + h_max
                z0 = model(atribute, t)

        if method == "Implicit_Euler":
            for _ in range(no_steps):
                t = t + h_max
                atribute = atribute + h_max
                z0 = model(atribute, t)
                integral += z0*h_max

        if method == "Trapezoid":
            for _ in range(no_steps):
                t = t + h_max            
                # atribute = atribute + h_max
                z1,h = model(atribute, t)
                integral += (z0+z1)*0.5*h_max
                z0 = z1

        if method == "Simpsons":
            z = []
            z.append(z0)
            for _ in range(no_steps):
                t = t + h_max
                atribute = atribute + h_max
                z0 = model(atribute, t)
                z.append(z0)
            integral = h_max/3*sum(z[0:-1:2] + 4*z[1::2] + z[2::2])

        if method == "Gaussian_Q":
            time_integral, weights = Gaussian_quadrature(no_steps, t0, t1)
            integral = 0
            for i in range(time_integral.shape[0]):
                t = time_integral[i]
                atribute = atribute + h_max
                z0 = model(atribute, t)
                integral += weights[i]*z0
            atribute = atribute + t1
            z0 = model(atribute, t1)

        return integral, z0, atribute

    integral_ = 0
    time_len = time.size(1)
    if atribute is None:
        atribute = torch.ones(in_size).reshape(1,-1).to(device)*farest_interevent
        atribute[:, 0] = 0
    z = torch.zeros(time.shape)
    z0, h = model(atribute, time[0, 0])
    z[:,0] = z0
    for i in range(time_len-1):
        start_index = i-in_size+1 if i-in_size+1 >= 0 else 0
        atribute[0, :i+1] = time[0, start_index:i+1, 0]
        z[:, i+1], h = model(atribute, time[:, i+1])
        integral_interval, z_, atribute = integral_solve(z0, time[0, :i], time[0, i], time[0, i+1],
                                                         atribute, no_steps=no_steps, h=h, method=method)
        # z_.register_hook(lambda grad: print("z_", grad, grad.size()))
        integral_ += integral_interval
        z[:, i+1] = z_
    return z, integral_

def loss(z, integral):
    ml = torch.sum(torch.log(z)) - integral
    return torch.neg(ml)


def fit(model, model_filepath, train_time, test_time, in_size, lr, device="cpu", method="Euler", no_steps=10, h=None, no_epoch=100, log=1,
        log_epoch=10, figpath=None, farest_interevent=-1e9):
    
    train_losses, test_losses = [], []
    #initiatilizing Weights
    init_loss = evaluate(model, train_time, in_size, no_steps=no_steps, device=device, h=h, method='Trapezoid',
                         farest_interevent=farest_interevent).cpu().data.numpy().flatten()[0]
    if np.isnan(init_loss) or np.isinf(init_loss):
        print(f"Init loss: {init_loss}. The model needs to be reinitialized.")
        return fit(model, train_time, test_time, in_size, lr, method, no_steps, h, no_epoch, log,
                   log_epoch, figpath, farest_interevent)

    list_1 = list(model.parameters())
    optimizer_1 = torch.optim.Adam(list_1, lr=lr)

    for e in range(no_epoch):
        model.train()
        optimizer_1.zero_grad()
        if method == "Analytical":
            z_, integral_ = model.integral_analytical(train_time)
        else:
            z_, integral_ = integral(model, train_time, in_size, no_steps=no_steps, device=device, h=h, method=method,
                                     farest_interevent=farest_interevent)
        train_loss = loss(z_, integral_)
        train_loss.backward()
        optimizer_1.step()

        train_losses.append(train_loss.cpu().data.numpy().flatten()[0])
        test_loss = evaluate(model, test_time, in_size, no_steps=no_steps, device=device, h=h, method='Trapezoid',
                             farest_interevent=farest_interevent)
        test_losses.append(test_loss.cpu().data.numpy().flatten()[0])
        torch.save(model, model_filepath)
        if e % log_epoch == 0 and log == 1:
            print(f"Epoch: {e}, train loss: {train_loss.cpu().data.numpy().flatten()[0]}, "
                  f"test loss: {test_loss.cpu().data.numpy().flatten()[0]}")
            
    #         if figpath:
    #             plt.clf()
    #             plt.plot(train_losses, color='skyblue', linewidth=2, label='train')
    #             plt.plot(test_losses, color='darkgreen', linewidth=2, linestyle='dashed', label="test")
    #             plt.legend(loc="upper right")
    #             plt.show()

    # if figpath:
    #     plt.clf()
    #     plt.plot(train_losses, color='skyblue', linewidth=2, label='train')
    #     plt.plot(test_losses, color='darkgreen', linewidth=2, linestyle='dashed', label="test")
    #     plt.legend(loc="upper right")
    #     plt.savefig(figpath)
    #     plt.show()

    return model


def search_for_optimal_no_steps(model, train_time, test_time, in_size, lr, device='cpu', improvement_threshold=0.01,
                                method="Trapezoid", log=1, farest_interevent=-1e9):
    no_steps_possible_values = [10, 100, 1000]
    losses_for_possible_no_steps = []
    for i, no_steps_value in enumerate(no_steps_possible_values):
        learned_model = fit(model, train_time, test_time, in_size, lr, method=method, device=device,
                            no_steps=no_steps_value, no_epoch=10, log=0, farest_interevent=farest_interevent)
        loss = evaluate(learned_model, train_time, in_size, no_steps=100, device=device, method=method, # todo: no_steps is fixed to 100 to avoid bias
                        farest_interevent=farest_interevent).cpu().data.numpy().flatten()[0]
        losses_for_possible_no_steps.append(loss)
        print(losses_for_possible_no_steps)
        if i != 0:
            improvement_percent = (losses_for_possible_no_steps[i-1] - loss) / losses_for_possible_no_steps[i-1]
            print(f"Improvement percent with {no_steps_value} steps: {improvement_percent}")
            if improvement_percent < improvement_threshold:
                optimal_no_steps = no_steps_possible_values[i-1]
                print(f"Improvement with {no_steps_value} steps is not significant. Search for optimal no_step is "
                      f"finished, no_steps={optimal_no_steps} is chosen.")
                return optimal_no_steps
            elif i == (len(no_steps_possible_values)-1):
                print(f"Improvement with the max no_steps={no_steps_value} is still significant. Search for optimal "
                f"no_step is finished after all possible no_steps and max value, no_steps={optimal_no_steps}, is "
                f"chosen.")
                return no_steps_value


def evaluate(model, time, in_size, no_steps=10, device='cpu', h=None, method="Trapezoid", farest_interevent=0):
    model.eval()
    z_, integral_ = integral(model, time, in_size, no_steps, device=device, h=h, method=method, farest_interevent=farest_interevent)
    loss1 = loss(z_, integral_)
    return loss1


def predict(model, time, in_size, atribute=None, farest_interevent=-1e9):
    model.eval()
    time_len = time.size(1)
    if atribute is None:
        atribute = torch.ones(in_size).reshape(1, -1) * farest_interevent
        atribute[:, 0] = 0
    z = torch.zeros(time.shape)
    z0 = model(atribute, time[0, 0])
    z[:, 0] = z0
    for i in range(time_len-1):
        start_index = i-in_size+1 if i-in_size+1 >= 0 else 0
        atribute[0, :i+1] = time[0, start_index:i+1, 0]
        z[:, i+1] = model(atribute, time[:, i+1])
    return z


def integral_testing_steps(model, time, in_size, no_steps, device, h=None, atribute=None, method="Trapezoid", farest_interevent=-1e9):

    def integral_solve(z0, time_to_t0, t0, t1, atribute, no_steps=10, h=None, method="Trapezoid"):
        
        if no_steps is not None:
            h_max = (t1 - t0)/no_steps  #h - lenght of segment
        elif h is not None:
            no_steps = math.ceil((t1 - t0)/h) #ako je zadat h(duzina intervala)
            h_max = (t1 - t0)/no_steps

        integral = 0
        t = t0

        def Gaussian_quadrature(n, lower_l, upper_l):
            m = (upper_l-lower_l)/2
            c = (upper_l+lower_l)/2
            [x,w] = p_roots(n+1)
            weights = m*w
            time_train_integral = m*x+c
            return time_train_integral, weights

        if method == "Euler":
            for _ in range(no_steps):
                integral += z0*h_max
                t = t + h_max
                # atribute = atribute + h_max
                z0 = model(atribute, t)

        if method == "Implicit_Euler":
            for _ in range(no_steps):
                t = t + h_max
                atribute = atribute + h_max
                z0 = model(atribute, t)
                integral += z0*h_max

        if method == "Trapezoid":
            for _ in range(no_steps):
                t = t + h_max            
                # atribute = atribute + h_max
                z1 = model(atribute, t)
                integral += (z0+z1)*0.5*h_max
                z0 = z1

        if method == "Simpsons":
            z = []
            z.append(z0)
            for _ in range(no_steps):
                t = t + h_max
                atribute = atribute + h_max
                z0 = model(atribute, t)
                z.append(z0)
            integral = h_max/3*sum(z[0:-1:2] + 4*z[1::2] + z[2::2])

        if method == "Gaussian_Q":
            time_integral, weights = Gaussian_quadrature(no_steps, t0, t1)
            integral = 0
            for i in range(time_integral.shape[0]):
                t = time_integral[i]
                atribute = atribute + h_max
                z0 = model(atribute, t)
                integral += weights[i]*z0
            atribute = atribute + t1
            z0 = model(atribute, t1)

        return integral, z0, atribute
    integral_ = 0
    time_len = time.size(1)
    if atribute is None:
        atribute = torch.ones(in_size).reshape(1,-1).to(device)*farest_interevent
        atribute[:, 0] = 0
    z = torch.zeros(time.shape)
    z0 = model(atribute, time[0, 0])
    z[:,0] = z0
    for i in range(time_len-1):
        start_index = i-in_size+1 if i-in_size+1 >= 0 else 0
        atribute[0, :i+1] = time[0, start_index:i+1, 0]
        z[:, i+1] = model(atribute, time[:, i+1])
        integral_interval, z_, atribute = integral_solve(z0, time[0, :i], time[0, i], time[0, i+1],
                                                         atribute, no_steps=no_steps, h=h, method=method)
        # z_.register_hook(lambda grad: print("z_", grad, grad.size()))
        integral_ += integral_interval
        z[:, i+1] = z_
        return z, integral_

def sliding_windows(datax, datay, seq_length):
    x = []
    y = []

    for i in range(len(datax) - seq_length - 1):
        _x = datax[i:(i + seq_length)]
#        _x = preprocessing.normalize(_x)
        _y = datay[i + seq_length]
        x.append(_x)
        y.append(_y)			

    return np.array(x), np.array(y)