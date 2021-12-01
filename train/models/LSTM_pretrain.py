import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import xlwt as xl
import pickle
import seaborn as sns
import time
import scipy as sc

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from sklearn import preprocessing
from pylab import rcParams
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import register_matplotlib_converters
from torch.utils.data import TensorDataset, DataLoader

#ploting settings
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#93D30C", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 14, 10
register_matplotlib_converters()

df = pd.read_csv('stan1_traka1_01012017.csv')
df = df['event_time']
values = np.array(df)

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
       

def sliding_window(df_, window_size :int, duzina_pozora_inteziteta : int): 
    '''
    X (Doalzak_1, dolaza_2 ... dolazak_n) --> Y(Uslovni intezitet dolazaka po minuti )
    
    1. Napravimo matricu sa momentima kada se desio point procesa u sekundi (ex. 0 0 0 0 1 0 0 0 0 ... --> dolazak se desio u 4. sekundi)
    2. U trenutku sledeceg dolaska treba nam intezitet_dolazaka (primer. prethodnih 30min(30*60s))
    3. Nalazimo lokaciju indexa trenutka t i računamo u nazad intezitete 
    
    Args:
        df_ (pd.Series) : df['event_time'] sa podatcima kada su se desili dogadjaji
        window_size (int) :  koliki prethodni period dolazaka posmatramo da bi predvideli intezitet 
        duzina_pozora_inteziteta (int) : uslovni intezitet je izrazen kao dolazaka po duzini_prozora_inteziteta
        
    Return:
        x (np.array) :
        y (np.array) :
    '''   
    values_in_minutes = np.array(df_)/60
    x = []
    y = []
    
    matrix = np.zeros(int(values[-1])+1)   
    for value in (values):
        matrix[int(value)] = 1 #pravimo matricu sa svim vrednostima u kojim su se trenutcima desili otkzi
        
    five_min = 5*60 / duzina_pozora_inteziteta #when multiplied with window_size(seq_len) returns values scaled to 5minutes    
    for i in range(len(values_in_minutes) - window_size):
        _x = values_in_minutes[i:i+window_size]
        trenutak_t = values_in_minutes[i+window_size]
        if trenutak_t*60 > duzina_pozora_inteziteta:
            lambda_y = sum(matrix[int(trenutak_t*60 - duzina_pozora_inteziteta) : int(trenutak_t*60)])*five_min
        else:
            lambda_y = sum(matrix[int(trenutak_t*60 - 15*60) : int(trenutak_t*60)])*((5*60)/(15*60))#scaled to 5min
        x.append(_x)
        y.append(lambda_y)
    return np.array(x), np.array(y).reshape(-1,1)

#x,y = sliding_window(df, 15, 30*60)
def plot_accuracy_loss(Loss): 
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.plot(Loss, 'b-')
    plt.ylabel('loss')
    plt.xlabel('epochs') 
    plt.title('training loss iterations')
    plt.show()

class LSTMNet(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, hidden_dim_fc1, output_dim, n_layers, drop_prob = 0):
        super(LSTMNet, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.hidden_dim_fc1 = hidden_dim_fc1
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim_fc1)
        self.fc2 = nn.Linear(hidden_dim_fc1, output_dim)
        self.relu = nn.ReLU()
        self.Batchn = nn.BatchNorm1d(hidden_dim_fc1)
    
    def forward(self, x, h):
        out, h = self.lstm(x, h)
        out = self.fc1(out)
        out = self.relu(out[:,-1])
        out = self.fc2(out)
        #out = self.relu(out)
        return out, h
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden

def train(train_loader, learn_rate, hidden_dim, number_of_layers, EPOCHS=300, model_type="GRU"):
    # Setting common hyperparameters
    input_dim = next(iter(train_loader))[0].shape[2]
    output_dim = 1
    n_layers = number_of_layers
    hidden_dim_fc1 = int(number_of_layers/2)
    # Instantiating the models
    if model_type == "LSTM":
        model = LSTMNet(input_dim, hidden_dim, hidden_dim_fc1, output_dim, n_layers)        
    model.to(device)

    # Defining loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    model.train()
    print("Starting Training of {} model".format(model_type))
    epoch_times = []
    LOSS = []
    # Start training loop
    for epoch in range(1, EPOCHS + 1):  
        h = model.init_hidden(batch_size)
        avg_loss = 0.
        counter = 0
        for x, label in train_loader:  #for i in range seq_len.
            counter += 1
            if model_type == "GRU" or model_type == "RNN":
                h = h.data
            else:
                h = tuple([e.data for e in h])
            model.zero_grad()
            out, h = model(x.to(device).float(), h)
            loss = criterion(out, label.to(device).float())
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            if counter % 100 == 0:
                print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter,
                                                                                           len(train_loader),
                                                                                           avg_loss / counter))
        LOSS.append((avg_loss / counter))
        print("Epoch {}/{} Done, Total Loss: {}".format(epoch, EPOCHS, avg_loss / len(train_loader)))
        if epoch % 30 == 0:
            plot_accuracy_loss(LOSS)
    #    print("Total Training Time: {} seconds".format(str(sum(epoch_times))))
    return model, LOSS


def evaluate(model, test_loader):
    model.eval()
    outputs = []
    targets = []
    loss = []
    h = model.init_hidden(testY.shape[0])
    for test_x, test_y in test_loader:
        out, h = model(test_x.to(device).float(), h)
        outputs.append(out.cpu().detach().numpy())
        targets.append(test_y.numpy())
    for i in range(len(test_x)):
        MSEloss = (outputs[0][i]-targets[0][i])**2
        loss.append(MSEloss)
    Loss = sum(loss)/len(loss)	
    print("Validation loss: {}".format(Loss))
    return outputs, targets, Loss

seq_length =  [ 50 ]
hidden_dim = [ 50  ]
number_of_layers = [ 2 ]

wb = xl.Workbook ()
ws1 = wb.add_sheet("RNN razultati")
ws1_kolone = ["Ime simulacije", "Training L","Validation Loss" ]
ws1.row(0).write(0, ws1_kolone[0])
ws1.row(0).write(1, ws1_kolone[1])
ws1.row(0).write(2, ws1_kolone[2])
ws2 = wb.add_sheet("GRU razultati")
ws2_kolone = ["Ime simulacije", "Training L","Validation Loss"]
ws2.row(0).write(0, ws1_kolone[0])
ws2.row(0).write(1, ws1_kolone[1])
ws2.row(0).write(2, ws1_kolone[2])
ws3 = wb.add_sheet("LSTM ruzultati")
ws3_kolone = ["Ime simulacije", "Training L","Validation Loss"]
ws3.row(0).write(0, ws1_kolone[0])
ws3.row(0).write(1, ws1_kolone[1])
ws3.row(0).write(2, ws1_kolone[2])
counter = 1
for seq_len in seq_length:
	for hid_dim in hidden_dim:
		for num_layers in number_of_layers:
			
			x,y = sliding_window(df, 15, 30*60)
			x = np.expand_dims(x, axis=-1)
			
			train_size = int(len(y) * 0.8)
			test_size = len(y) - train_size
			
			dataX = np.array(x)
			dataY = np.array(y)
			trainX = np.array(x[0:train_size])
			
			trainY = np.array(y[0:train_size])
			
			testX = np.array(x[train_size:len(x)])
			testY = np.array(y[train_size:len(y)])
			
			#Data loader
			batch_size = 32	
			train_data = TensorDataset(torch.from_numpy(trainX), torch.from_numpy(trainY))
			train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size, drop_last=True)
			test_data = TensorDataset(torch.from_numpy(testX), torch.from_numpy(testY))
			test_loader = DataLoader(test_data, shuffle=False, batch_size=testY.shape[0], drop_last=True)
			lr = 0.0001

            
			#Training and Validating LSTM_model
            
			lstm_model, lstm_training_loss = train( train_loader, lr, hid_dim, num_layers, model_type="LSTM")
			lstm_outputs, lstm_targets, lstm_test_loss = evaluate(lstm_model, test_loader)
            
			simulation_name = 'Pre_trained_LSTM_point_process' 
			pathLSTM =  simulation_name + '.pt'
            
						#LSTM
			ws3.row(counter).write(0, simulation_name + "_" +'LSTM')
			ws3.row(counter).write(1, lstm_training_loss[-1])
			ws3.row(counter).write(2, int(lstm_test_loss[-1]))

			#save model parametre LSTM
			torch.save(lstm_model, pathLSTM)
			
wb.save(simulation_name + ".xls")
