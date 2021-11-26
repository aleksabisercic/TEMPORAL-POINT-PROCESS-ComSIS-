# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 14:54:46 2019

@author: Andri
"""
import models as ml
import torch
import numpy as np
import pandas as pd


class Simulation:

    def __init__(self, fun, time_upper, time_lower):
        self.time_upper = time_upper
        self.time_lower = time_lower
        self.mod = fun

    def step_simulation(self, dataset='autoput', atributes=None, no_steps_max=500):
        n = 0
        m = 0
        tn = [0]
        sm = self.time_lower
        if dataset == 'ski':
            train_df = pd.read_csv('1.Proba.csv')
        elif dataset == 'autoput':
            train_df = pd.read_csv('stan1_traka1_01012017.csv')

        while sm < self.time_upper:
            time = np.linspace(sm, self.time_upper, no_steps_max)
            time = torch.tensor(time).type('torch.FloatTensor').reshape(-1, 1)
            if dataset == 'ski':
                atributes = torch.tensor(train_df.event_time.values[
                                         int(len(train_df.event_time.values) * 0.97 - 500):int(
                                             len(train_df.event_time.values) * 0.97)]).type(
                    'torch.FloatTensor').reshape(-1, 1)
            else:
                atributes = torch.tensor(train_df.event_time.values[
                                         int(len(train_df.event_time.values) * 0.8 - 500):int(
                                             len(train_df.event_time.values) * 0.8)]).type('torch.FloatTensor').reshape(
                    -1, 1)
            lamb_max = self.mod.predict_sim(atributes, time).max().data.numpy()
            if lamb_max.data == np.inf:
                lamb_max = 1e9
            u = np.random.rand()
            w = - np.log(u) / lamb_max
            sm = sm + w
            D = np.random.rand()
            lamb = self.mod.predict_sim(atributes[0], torch.tensor(sm)).data.numpy()
            if sm > self.time_upper: break
            if D * lamb_max < lamb.data:
                tn.append(sm)
                n += 1
            m += 1
        return tn

    def simulate(self, no_simulation=1):
        simulation = [Simulation.step_simulation(self) for _ in range(no_simulation)]
        return simulation
