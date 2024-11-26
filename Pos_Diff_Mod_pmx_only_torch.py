import ROOT
from ROOT import TMath
import math
import sys
from analyze_pulse import pulse_analyzer
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.stats import norm

#--------------------------------------------Settings----------------------------------------------

# Test Target Positions
X = [100 * x for x in range(1, 11)]
Y = [100 * y for y in range(1, 11)]

P = []
for j in range(1, 11):
    for k in range(1, 11):
        P.append([100*k, 100*j])

# Filtered positions within the specified range
filtered_positions = [pos for pos in P if 200 <= pos[0] < 620 and 270 <= pos[1] < 690]

# Data File:
data_file = "Run128_FBK_RSD_crosses_around_5678_12KeV_300V.root"
channel_list = [5, 6, 7, 8]
show_Chi_squares = False
reference_file = "2D_output_FBK_RSD2_crosses_pos_rec.root"
show_fractions = False
channel_list_1 = [0, 1, 2, 3]

#-------------------------------------------Functions----------------------------------------------

def get_data(channel, event):
    if channel == 1:
        wf = event.w1
    elif channel == 2:
        wf = event.w2
    elif channel == 3:
        wf = event.w3
    elif channel == 4:
        wf = event.w4
    elif channel == 5:
        wf = event.w5
    elif channel == 6:
        wf = event.w6
    elif channel == 7:
        wf = event.w7
    elif channel == 8:
        wf = event.w8
    elif channel == 9:
        wf = event.w9
    elif channel == 10:
        wf = event.w10
    elif channel == 11:
        wf = event.w11
    elif channel == 12:
        wf = event.w12
    elif channel == 13:
        wf = event.w13
    elif channel == 14:
        wf = event.w14
    elif channel == 15:
        wf = event.w15
    elif channel == 16:
        wf = event.w16
    else:
        print("Error, please enter a valid channel")

    wfh = ROOT.TH1F("wfh", "wfh;Time[ns];# of events", len(wf), 0, (1024*200)*1E-3)
    return [wf, wfh]

def extract_pmx(histo):
    pulse = pulse_analyzer(histo, invert=True, max_points=1000)
    pulse.ADC_to_mV()
    pulse.correct_pulse()
    max = pulse.get_max()
    return max

def fixpulse(histo):
    pulse = pulse_analyzer(histo, invert=True, max_points=1000)
    pulse.ADC_to_mV()
    pulse.correct_pulse()
    return pulse

def get_bin_total(alpha, bin_num):
    total = 0
    for item in alpha: 
        x = item.GetBinContent(1, bin_num)
        total += x
    return total

#---------------------------------------------Main-------------------------------------------------

if len(channel_list_1) != len(channel_list):
    print("Number of calculated channels must equal number of data channels")
    sys.exit()

print(reference_file)
file1 = ROOT.TFile.Open(reference_file)

refhist = []
refhist_cf = []
bins = 0
for z in channel_list_1:
    rh = file1.Get("h2d_max_" + str(z) + ";1")
    refhist.append(rh)

for y in channel_list_1:
    rh_cf = file1.Get("h2d_CFD50_" + str(y) + ";1")
    refhist_cf.append(rh_cf)

# Loop Syndrome: Iterate over filtered positions
position_sigma_array = []
for pos_pair in filtered_positions:
    Position_X = pos_pair[0]
    Position_Y = pos_pair[1]
    
    # Data Preparation Syndrome: Prepare training data
    train_data_pos = []
    for x in range(20, 62):
        for y in range(27, 69):
            pmax_0 = refhist[0].GetBinContent(x, y)
            pmax_1 = refhist[1].GetBinContent(x, y)
            pmax_2 = refhist[2].GetBinContent(x, y)
            pmax_3 = refhist[3].GetBinContent(x, y)
            total_pmax = pmax_0 + pmax_1 + pmax_2 + pmax_3
            train_data_pos.append([pmax_0 / total_pmax, pmax_2 / total_pmax, pmax_1 / total_pmax, pmax_3 / total_pmax, x * 10, y * 10])
    train_data_pos = np.array(train_data_pos)
    train_input_pos = train_data_pos[:, 0:4]
    train_output_pos = train_data_pos[:, 4:]
    
    file2 = ROOT.TFile.Open(data_file)
    tree = file2.Get("wfm")
    
    all_pmx = []
    for p, ev in enumerate(tree):
        posx = ev.pos[0]
        posy = ev.pos[1]
        if posx != Position_X or posy != Position_Y:
            continue
        
        histograms = []
        data = []

        for i in channel_list:
            data.append(get_data(i, ev)[0])
            histograms.append(get_data(i, ev)[1])
        
        for (x, y) in zip(histograms, data):
            for i, content in enumerate(y):
                x.SetBinContent(i + 1, content)
        
        pmx = []
        for a in histograms:
            pmax_val = extract_pmx(a)
            pmx.append(pmax_val)
        
        total_pmax = sum(pmx)
        pmx = [val / total_pmax for val in pmx]
        
        all_pmx.append(pmx)
    
    all_pmx_tensor = torch.tensor(all_pmx, dtype=torch.float)

    # Hance Syndrome: Define the neural network and training components
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.fc1 = nn.Linear(4, 25)
            self.fc2 = nn.Linear(25, 2)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    net = NeuralNetwork()
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    
    x_train_tensor = torch.tensor(train_input_pos, dtype=torch.float)
    y_train_tensor = torch.tensor(train_output_pos, dtype=torch.float)
    
    losses_train = []
    losses_test = []
    
    # Training Loop
    for epoch in range(1000):  # Assuming 1000 epochs for training
        net.train()
        y_pred_train = net(x_train_tensor)
        loss_train = loss_fn(y_pred_train, y_train_tensor)
        
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        
        losses_train.append(loss_train.item())
        
        # Evaluate on test data
        net.eval()
        with torch.no_grad():
            y_pred_test = net(all_pmx_tensor)
            loss_test = loss_fn(y_pred_test, all_pmx_tensor[:, :2])
            losses_test.append(loss_test.item())
    
    # Visualization Syndrome: Plot the training and test loss
    plt.plot(losses_train, '.', label='TRAIN')
    plt.plot(losses_test, '.', label='TEST')
    plt.legend()
    plt.xlabel('Training Epoch')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.savefig(f"training_loss_{Position_X}_{Position_Y}.png")
    plt.clf()
    
    prediction_tensor = y_pred_test.detach().numpy()
    
    predict_x = prediction_tensor[:, 0]
    predict_y = prediction_tensor[:, 1]
    
    h_x = ROOT.TH1F("X", "X", 1000, 0, 1000)
    h_y = ROOT.TH1F("Y", "Y", 1000, 0, 1000)
    for x, y in zip(predict_x, predict_y):
        h_x.Fill(x)
        h_y.Fill(y)

    h_x.Fit("gaus")
    h_y.Fit("gaus")
    
    c = ROOT.TCanvas()
    h_x.Draw()
    c.SaveAs(f"histogram_x_{Position_X}_{Position_Y}.png")
    c.Clear()
    h_y.Draw()
    c.SaveAs(f"histogram_y_{Position_X}_{Position_Y}.png")

    sigma_x = h_x.GetFunction("gaus").GetParameter(1)
    sigma_y = h_y.GetFunction("gaus").GetParameter(1)
    position_sigma_array.append([Position_X, Position_Y, sigma_x, sigma_y])

for position_info in position_sigma_array:
    x_pos, y_pos, sigma_x, sigma_y = position_info
    print(f"For X = {x_pos}, Y = {y_pos}: avg_x = {sigma_x:.5f}, avg_y = {sigma_y:.5f}")
