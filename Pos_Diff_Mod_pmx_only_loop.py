import ROOT
from ROOT import TMath
import math
import sys
from analyze_pulse import pulse_analyzer
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import norm

#--------------------------------------------Settings----------------------------------------------

# Test Target Positions
X = [100 * x for x in range(1, 11)]
Y = [100 * y for y in range(1, 11)]

P = []
for j in range(1, 11):
    for k in range(1, 11):
        P.append([100 * k, 100 * j])

# Filtered positions within the specified range
filtered_positions = [pos for pos in P if 200 <= pos[0] < 620 and 270 <= pos[1] < 690]

# Open a PdfPages object to save all plots into a single PDF
with PdfPages('master_plots.pdf') as pdf:
    # Data File:
    for pos in filtered_positions:
        data_file = "Run128_FBK_RSD_crosses_around_5678_12KeV_300V.root"
        channel_list = [5, 6, 7, 8]
        Position_X = pos[0]
        Position_Y = pos[1]

        show_Chi_squares = False

        # Reference File:
        reference_file = "2D_output_FBK_RSD2_crosses_pos_rec.root"
        show_fractions = False
        channel_list_1 = [0, 1, 2, 3]

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
            wfh = ROOT.TH1F("wfh", "wfh;Time[ns];# of events", len(wf), 0, (1024 * 200) * 1E-3)
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

        # Training model
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

        train_input = train_input_pos
        train_output = train_output_pos

        model = MLPRegressor(max_iter=10000, random_state=52)
        model.fit(train_input, train_output)
        joblib.dump(model, 'trained_model_pmx_only.pkl')

        model = joblib.load('trained_model_pmx_only.pkl')

        if show_fractions:
            for fr in refhist:
                c = ROOT.TCanvas()
                fr.Draw("lego1")
                cntrl = str(input())
                if cntrl == "q":
                    sys.exit()
                elif cntrl == "s" or cntrl == "S":
                    break

        file2 = ROOT.TFile.Open(data_file)
        tree = file2.Get("wfm")

        all_prediction = []
        all_pmax = []
        all_positions = []
        pos_pred_dict = {}
        predict_x = []
        predict_y = []
        pos_input_x = []
        pos_input_y = []
        positionsarray = []

        for p, ev in enumerate(tree):
            if len(channel_list) > 4:
                print("Too many Channels")
                break

            posx = ev.pos[0]
            posy = ev.pos[1]
            positions = (posx, posy)
            all_positions.append(positions)
            if positions not in pos_pred_dict:
                pos_pred_dict.update({positions: []})

            if positions not in positionsarray:
                positionsarray.append(positions)

            if posx != Position_X or posy != Position_Y:
                continue
            else:
                pos_input_x.append(posx)
                pos_input_y.append(posy)
                histograms = []
                data = []

                for i in channel_list:
                    data.append(get_data(i, ev)[0])
                    histograms.append(get_data(i, ev)[1])

                for (x, y) in zip(histograms, data):
                    for i, content in enumerate(y):
                        x.SetBinContent(i + 1, content)

                pmx = []
                cfd = []
                total_pmax = 0
                total_CFD = 0
                for a in histograms:
                    pmax_val = extract_pmx(a)
                    total_pmax += pmax_val
                    pmx.append(pmax_val)

                for i in range(len(pmx)):
                    pmx[i] = pmx[i] / total_pmax

                pmx = np.array(pmx)
                pmx = [pmx]
                prediction = model.predict(pmx)[0]
                pos_pred_dict[positions].append(prediction)

                predict_x.append(prediction[0])
                predict_y.append(prediction[1])

        pos_input_x_arr = np.array(pos_input_x)
        pos_input_y_arr = np.array(pos_input_y)

        fig, ax = plt.subplots(2)
        test_position = (Position_X, Position_Y)

        # Plotting the 2D scatter plot with average position and σ
        plt.figure(figsize=(10, 8))
        plt.scatter(predict_x, predict_y, color='blue', label='Predicted Coordinates')
        plt.axhline(y=np.mean(predict_y), color='r', linestyle='--', label=f'Avg. Y: {np.mean(predict_y):.2f}')
        plt.axvline(x=np.mean(predict_x), color='g', linestyle='--', label=f'Avg. X: {np.mean(predict_x):.2f}')
        plt.xlabel('Predict X')
        plt.ylabel('Predict Y')
        plt.title(f'2D Position Plot (Test Position: {test_position})\n'
                  f'Avg. X: {np.mean(predict_x):.2f}, Avg. Y: {np.mean(predict_y):.2f}')
        plt.legend()
        plt.grid(True)

        # Save the 2D position plot to the PDF
        pdf.savefig()
        plt.close()

        # Create histograms
        h_x = ROOT.TH1F("X", "X Position Histogram", 1000, 0, 1000)
        h_y = ROOT.TH1F("Y", "Y Position Histogram", 1000, 0, 1000)

        for x, y in zip(predict_x, predict_y):
            h_x.Fill(x)
            h_y.Fill(y)

        # Fit histograms to a Gaussian and extract parameters
        h_x.Fit("gaus")
        h_y.Fit("gaus")

        mean_x = h_x.GetFunction("gaus").GetParameter(1)
        std_x = h_x.GetFunction("gaus").GetParameter(2)
        mean_y = h_y.GetFunction("gaus").GetParameter(1)
        std_y = h_y.GetFunction("gaus").GetParameter(2)

        # Convert ROOT histograms to numpy arrays for Matplotlib
        bin_edges_x = np.array([h_x.GetBinLowEdge(i+1) for i in range(h_x.GetNbinsX()+1)])
        bin_values_x = np.array([h_x.GetBinContent(i+1) for i in range(h_x.GetNbinsX())])

        bin_edges_y = np.array([h_y.GetBinLowEdge(i+1) for i in range(h_y.GetNbinsX()+1)])
        bin_values_y = np.array([h_y.GetBinContent(i+1) for i in range(h_y.GetNbinsX())])

        # Plot histograms with Gaussian fits
        plt.figure(figsize=(8, 6))
        plt.hist(bin_edges_x[:-1], bins=bin_edges_x, weights=bin_values_x, alpha=0.6, color='g', label='Histogram')
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mean_x, std_x) * h_x.GetEntries() * (bin_edges_x[1] - bin_edges_x[0])
        plt.plot(x, p, 'k', linewidth=2, label='Gaussian Fit')
        plt.title(f'X Histogram (Test Position: {test_position})\n'
                  f'Avg. X: {mean_x:.2f}, X σ: {std_x:.2f}')
        plt.xlabel('X')
        plt.ylabel('Counts')
        plt.legend()

        # Save the X histogram to the PDF
        pdf.savefig()
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.hist(bin_edges_y[:-1], bins=bin_edges_y, weights=bin_values_y, alpha=0.6, color='g', label='Histogram')
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mean_y, std_y) * h_y.GetEntries() * (bin_edges_y[1] - bin_edges_y[0])
        plt.plot(x, p, 'k', linewidth=2, label='Gaussian Fit')
        plt.title(f'Y Histogram (Test Position: {test_position})\n'
                  f'Avg. Y: {mean_y:.2f}, Y σ: {std_y:.2f}')
        plt.xlabel('Y')
        plt.ylabel('Counts')
        plt.legend()

        # Save the Y histogram to the PDF
        pdf.savefig()
        plt.close()

        # Print Gaussian fit parameters
        print(f'Test Position: {test_position}')
        print(f'X Gaussian Fit Mean: {mean_x:.2f}, Std Dev: {std_x:.2f}')
        print(f'Y Gaussian Fit Mean: {mean_y:.2f}, Std Dev: {std_y:.2f}')

