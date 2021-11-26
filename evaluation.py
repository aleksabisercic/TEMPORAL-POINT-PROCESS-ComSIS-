# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 14:54:46 2019

@author: Andri
"""

import torch
import numpy as np
import pandas as pd
import pickle
from simulation import Simulation


def evaluate(train_df: pd.DataFrame, path, bin_size, model_name):
    """
    Get events per bin for each simulation for a given model
    
    Args:
        train_df: pd.DataFrame
            pd.DataFrame with ground truth data
        path (pd.DataFrame or str)
            path to pre-trained model
        bin_size (int)
            bin_size or window_size
        model_name (str)
            name of the model being evaluated
    Return:
        saved df (pd.DataFrame)
            dataframe with:
                - number of events by bin
                - MAE by bin
    """

    train = train_df[int(len(train_df.event_time.values) * 0.75):]
    train = (train['event_time'] - train.event_time.min()) / 60
    train = train.to_frame()
    train.reset_index(inplace=True)

    if type(path) == str:
        simulated = pd.read_csv(path)
    else:
        simulated = path

    bins = list(np.arange(0, train.event_time.max() + bin_size, bin_size))
    bins[0] = -1
    labels = list(np.arange(1, len(bins)))

    train['5m_binned'] = pd.cut(train['event_time'], bins=bins, labels=labels)
    binned_train = train['5m_binned'].value_counts(sort=False).reset_index()
    binned_train.columns = ['bin', 'train']

    binned_train.head()

    def get_bin_for_each_simulation(simulated_df, row):
        simulated_temp = pd.cut(simulated_df.iloc[row].T.dropna().values, bins=bins, labels=labels)
        simulation_by_bin = simulated_temp.value_counts().reset_index()
        simulation_by_bin.columns = ['bin', 'simulated']

        return simulation_by_bin.set_index('bin')

    # get bins by simulation
    binned_simulated = [get_bin_for_each_simulation(simulated, row)
                        for row in range(simulated.shape[0])]

    # concat into single df
    binned = pd.concat(binned_simulated + [binned_train.set_index('bin')], axis=1)

    binned.rename(columns={'train': 'Groun_truth'}
                  ).to_csv(f'Results/{model_name}_{bin_size}.csv')

    def get_mae_data(binned_data):
        """
        Generate MAE for window_size = bin_size, for each simulation

        Args:
        binned_data (pd.DataFrame),
            sim and real data binned where window size is bin_size


        Return:
        df (pd.DataFrame)
            dataframe MAE per window for each simulation

        """
        df_mae = binned_data.copy()
        for simulation_number in range(df_mae.shape[1] - 1):
            df_mae.iloc[:, simulation_number] = np.absolute(df_mae.iloc[:, simulation_number] - df_mae['train'])
        return df_mae

    df_mae = get_mae_data(binned).drop(columns=['train'])

    df_mae.to_csv(f'Results/MAE_DF_{model_name}_{bin_size}.csv')

    def get__total_avrage_mae(df_mae):
        avrage_error = df_mae.mean().mean()
        print(f'Evrage for this sim is MAE is {avrage_error}')

    get__total_avrage_mae(df_mae)


def get_simulation_times(model_filepath: str = "Hawks_Trapezoid_cpu.torch", no_sim: int = 1, time_upper: int = 500):
    """
    Levraging pretrained model for (CIF) generate event times 
    via Ogata’s modified thinning algorithm for n (no_sim) simulations
    
    Args:
        model_filepath (str), defult training_loss_5_116_cpu.torch
            path to pre-trained model
        no_sim (int), defult 1
            number of simulations to run
        time_upper: (int), default  500

    Return:
        df (pd.DataFrame) 
            dataframe with event_times for no_sim simulations
    """

    try:
        # if device has no cuda
        mod = torch.load(model_filepath, map_location=torch.device('cpu'))
    except:
        with open(model_filepath, 'rb') as handle:
            mod = pickle.load(handle)

    for param in mod.parameters():
        print(param)

    sim_model = Simulation(fun=mod, time_upper=time_upper, time_lower=0)
    simulation = sim_model.simulate(no_simulation=no_sim)
    print(simulation)

    simulated_df = pd.DataFrame(simulation)

    bin_sizes = [5, 10, 15]

    try:
        model_name = model_filepath.split('\\')[1].split('_')[0]
    except IndexError:
        model_name = model_filepath.split('/')[1].split('_')[0]

    real_data = pd.read_csv('data/stan1_traka1_01012017.csv')

    for bin_size in bin_sizes:
        print(f'{model_name} for bin_size: {bin_size} days')
        evaluate(path=simulated_df,
                 bin_size=bin_size,
                 train_df=real_data,
                 model_name=model_name)
        print('\n')

