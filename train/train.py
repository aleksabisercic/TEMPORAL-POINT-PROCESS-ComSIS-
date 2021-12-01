import os
import pandas as pd
import pickle
import numpy as np
# from train import NNTraining as train
import time
import torch
from pathlib import Path

from models import LSTMPointProcess, HawkesTPP, PoissonPolynomialTPP, GausTPP, PoissonTPP
from src import BaselineTraining as train

if __name__ == "__main__":

    # Pandas option
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    project_dir = str(Path(__file__).parent.parent)

    # CUDA setup
    print(f'Cuda is available: {torch.cuda.is_available()}')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        device_id = torch.device(device)
        torch.cuda.set_device(device_id)
        print(f"Current active cuda: {torch.cuda.current_device()}")

    # Set respective directory
    data_folder = project_dir + '/data/autoput/'
    train_df = pd.read_csv('stan1_traka1_01012017.csv')
    test_df = pd.read_csv('stan1_traka1_01012017.csv')
    train_time = torch.tensor(train_df.event_time.values[:int(len(train_df.event_time.values) * 0.8)]).type(
        'torch.FloatTensor').reshape(1, -1, 1).to(device)
    test_time = torch.tensor(test_df.event_time.values)[int(len(train_df.event_time.values) * 0.8):].type(
        'torch.FloatTensor').reshape(1, -1, 1).to(device)
    in_size = 10
    out_size = 1

    rules = ['Trapezoid', 'Implicit_Euler', 'Simpsons']
    farest_interevent = 0
    no_epochs = 200
    for rule_ in rules:
        rule = rule_
        model = HawkesTPP(in_size, out_size).to(device)

        no_steps = 200
        model_name = 'LSTMPointProcess'
        t0 = time.time()
        model_filepath = f"{model_name}_{rule}.torch"
        model = train.fit(model, model_filepath, train_time, test_time, in_size, lr=0.001,
                          no_epoch=no_epochs, device=device, no_steps=no_steps, method=rule, log_epoch=1,
                          figpath=f"{project_dir}/img/autoput/{model_name}_train.png",
                          farest_interevent=farest_interevent)

        loss_on_train = train.evaluate(model, train_time, in_size, device=device, method=rule,
                                       farest_interevent=farest_interevent)
        loss_on_test = train.evaluate(model, test_time, in_size, device=device, method=rule,
                                      farest_interevent=farest_interevent)
        print(f"Model: {model_name}. Loss on train: {str(loss_on_train.cpu().data.numpy().flatten()[0])}, "
              f"loss on test: {str(loss_on_test.cpu().data.numpy().flatten()[0])}")

        pickle.dump(model, open(model_filepath, 'wb'))
        torch.save(model, model_filepath)
        ls = [loss_on_train.cpu().data.numpy().flatten()[0], loss_on_test.cpu().data.numpy().flatten()[0]]

        model_filepath_ = f"Test_simulations/Log_loss/{model_name}_{rule}_LOSS.npy"

        df = pd.DataFrame(np.array(ls).reshape(-1, 2), columns=['Train_loss', 'Test_loss'])
        df.to_csv(model_filepath_ + ".csv")
