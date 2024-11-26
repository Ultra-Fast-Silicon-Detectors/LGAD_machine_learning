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
from scipy.stats import norm

#--------------------------------------------Settings----------------------------------------------

# Test Target Positions
X = [100 * x for x in range(1, 11)]
Y = [100 * y for y in range(1, 11)]

P = []
for j in range(1, 11):
    for k in range(1,11):
        P.append([100*k,100*j])


# Filtered positions within the specified range
filtered_positions = [pos for pos in P if 200 <= pos[0] < 620 and 270 <= pos[1] < 690]
#[[200, 300], [300, 300], [400, 300], [500, 300], [600, 300], [200, 400], [300, 400], 
# [400, 400], [500, 400], [600, 400], [200, 500], [300, 500], [400, 500], [500, 500], 
# [600, 500], [200, 600], [300, 600], [400, 600], [500, 600], [600, 600]]

#Data File:
data_file = "Run128_FBK_RSD_crosses_around_5678_12KeV_300V.root"

channel_list = [5,6,7,8]

Position_X = 300

Position_Y = 400


show_Chi_squares = False

#Reference File:
reference_file = "2D_output_FBK_RSD2_crosses_pos_rec.root"

show_fractions = False
channel_list_1 = [0,1,2,3]


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

    wfh = ROOT.TH1F("wfh", "wfh;Time[ns];# of events", len(
        wf), 0, (1024*200)*1E-3)

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


#def minimum(graph):
#    m = graph.GetX(graph.GetMinimum())
#    return m

#def prepare_reference(histograms):
#    f = ROOT.TFile.Open(ref)
    #for hist in histograms:


#    data = []
#    for x in range(20,62):
#        for y in range(27,69):
#            data.append([histograms[0].GetBinContent(x,y), histograms[2].GetBinContent(x,y), 
#            histograms[1].GetBinContent(x,y), histograms[3].GetBinContent(x,y),x*10,y*10])
#    data = np.array(data)

#---------------------------------------------Main-------------------------------------------------

if len(channel_list_1) != len(channel_list):
    print("Number of calculated channels must equal number of data channels")
    sys.exit()
#exp
print(reference_file)
file1 = ROOT.TFile.Open(reference_file)

refhist = []
refhist_cf = []
bins = 0
for z in channel_list_1:
    rh = file1.Get("h2d_max_" + str(z)+";1")
    refhist.append(rh)


for y in channel_list_1:
    rh_cf = file1.Get("h2d_CFD50_" + str(y)+";1")
    refhist_cf.append(rh_cf)


## Training model 
train_data_pos = []

#print(refhist)
for x in range(20,62):
    for y in range(27,69):
        pmax_0 = refhist[0].GetBinContent(x,y)
        pmax_1 = refhist[1].GetBinContent(x,y)
        pmax_2 = refhist[2].GetBinContent(x,y)
        pmax_3 = refhist[3].GetBinContent(x,y)
        total_pmax = pmax_0 + pmax_1 + pmax_2 + pmax_3
        train_data_pos.append([pmax_0/total_pmax, pmax_2/total_pmax, 
        pmax_1/total_pmax, pmax_3/total_pmax,x*10,y*10])
train_data_pos = np.array(train_data_pos)
train_input_pos = train_data_pos[:,0:4]
train_output_pos = train_data_pos[:,4:]



train_input = train_input_pos
train_output = train_output_pos


##model = MultiOutputRegressor(GradientBoostingRegressor(random_state = 1, max_depth = 8))
model = MLPRegressor(max_iter = 10000, random_state=52)
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

#frachist = []
#prepdata = []
#for y in refhist:
    #fractions = ROOT.TH1F("Fractions", "Fractions Data", bins, 0, bins+1)
    #for i in range(1, bins+1):
        #percent = y.GetBinContent(1, i)/get_bin_total(refhist, i)
        #fractions.SetBinContent(i, percent)
    #frachist.append(fractions)

"""if show_fractions and cntrl != "S":
    print("drawing fractions histogram")
    c = ROOT.TCanvas()
    frachist[0].Draw()
    for col, h in enumerate(frachist):
        h.Draw("same")
        if col == 0:
            h.SetLineColor(ROOT.kRed)
        elif col == 1:
            h.SetLineColor(ROOT.kGreen)
        elif col == 2:
            h.SetLineColor(ROOT.kViolet)
        elif col == 3:
            h.SetLineColor(ROOT.kOrange)
        elif col == 4:
            h.SetLineColor(ROOT.kCyan)
        elif col == 5:
            h.SetLineColor(ROOT.kBlack)
    input()
"""



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

    
    #Set position:
    posx = ev.pos[0]
    posy = ev.pos[1]
    positions = (posx, posy)
    all_positions.append(positions)
    if positions not in pos_pred_dict:
        pos_pred_dict.update({positions: []})

    
    #print(positions)
    if(positions not in positionsarray):
        positionsarray.append(positions)
    
    if posx != Position_X or posy != Position_Y:
        
        continue
    else:
        pos_input_x.append(posx)
        pos_input_y.append(posy)
        histograms = []
        data = []

        #histograms for channels
        for i in channel_list:
            data.append(get_data(i, ev)[0])
            histograms.append(get_data(i, ev)[1])
        
        #print(positions, histograms)
        #Fill all histograms with data:
        for (x, y) in zip(histograms, data):
            for i, content in enumerate(y):
                x.SetBinContent(i+1, content)

        #print(positions, data)
        #Fix all histograms and get pmax:
        
        pmx = []
        cfd = []
        total_pmax = 0
        total_CFD = 0
        for a in histograms:
            #c = ROOT.TCanvas()
            #a.Draw()
            #c.Draw()
            #input()
            #print(pmax(a))
            pmax_val = extract_pmx(a)
            total_pmax += pmax_val
            pmx.append(pmax_val)

        #print(pmx)
        for i in range(len(pmx)):
            pmx[i] = pmx[i]/total_pmax

        
        pmx = np.array(pmx)
        pmx =[pmx]
        prediction = model.predict(pmx)[0]
        #print(prediction)
        pos_pred_dict[positions].append(prediction)
        
        
        #print(prediction[0])
        predict_x.append(prediction[0])
        predict_y.append(prediction[1])



        #print(posx, posy)
        
        #if(posx == 0 and posy ==0 ):
            #all_prediction.append(prediction)

        #print("prediction", prediction)
        #print("actual", posx, posy)
        


        #pmx.append(pmax(a))

        
        #print(histograms)
        #Get pmax percentages:
        #total = sum(pmx)
        #pfrac = []
        #for t in range(len(pmx)):
            #pfrac.append(pmx[t]/total)

        #Xisquare = ROOT.TGraph(bins)
        #Xisquare.SetTitle("Xi Square vs Position")
        #Xisquare.GetYaxis().SetTitle("X^2")
        #Xisquare.GetXaxis().SetTitle("Position[µm]")

        #bs = []
        #xs = []

        #Fill Xi Square Graph
        #for b in range(bins):
            #Xi = 0
            #if b < 35 or b > 70: continue
        # for n in range(len(channel_list)):
                #x = (pfrac[n] - frachist[n].GetBinContent(b))
                #Xi += x*x
            #Xisquare.AddPoint(b*bin_size, Xi)
            #bs.append(b*bin_size)
            #xs.append(Xi)

        #if p % 50 == 0:
            #print(p)

        #func = ROOT.TF1("func", "-pol2",
                        #bs[xs.index(min(xs))] - 80, bs[xs.index(min(xs))] + 80)
        #Xisquare.Fit(func, "QR")

        if show_Chi_squares:
            c = ROOT.TCanvas()
            Xisquare.Draw()
            print("Minimum: ", min(xs))
            print("Reconstructed point bin number: ", bs[xs.index(min(xs))])
            print("Reconstructed point: ", func.GetMinimumX())
            c.Draw()
            cmd = str(input())
            if cmd == "q":
                sys.exit()
            elif cmd == "s":
                show_Chi_squares = False

    #g.Fill(func.GetMinimumX())  # bs[xs.index(min(xs))]) #func.GetMinimumX())
#print(positionsarray)
#p = ROOT.TH1F("pmax","pmax", 100,0,1600)
#print(len(pmx))


        #for i in pos_pred_dict[key]:
            #print(i)

                
pos_input_x_arr = np.array(pos_input_x)
pos_input_y_arr = np.array(pos_input_y)

"""
mse_x = np.mean((pos_input_x_arr - predict_x)**2)
mse_y = np.mean((pos_input_y_arr - predict_y)**2)
print(f"Mean Squared Difference for x: {mse_x}")
print(f"Mean Squared Difference for y: {mse_y}")

"""




fig, ax = plt.subplots(2)


test_position = (Position_X, Position_Y)

# Plotting
plt.figure(figsize=(8, 6))
plt.scatter(predict_x, predict_y, color='blue', label='Predicted Coordinates')

# Adding labels and title
plt.xlabel('predict_x')
plt.ylabel('predict_y')
plt.title(f'2D Position Plot (Test Position: {test_position})')
plt.legend()

# Save the plot as an image file
plt.savefig('2d_position_plot.png')


h = ROOT.TH1F("X","X", 1000,0,1000)
h2 = ROOT.TH1F("Y", "Y", 1000,0,1000)
for x,y in zip(predict_x,predict_y):
    h.Fill(x)
    h2.Fill(y)
h.Fit("gaus")
h2.Fit("gaus")
c = ROOT.TCanvas()
h.Draw()
c.SaveAs("histogram_x" + "_pmx_only" + ".png")

c.Clear()  # Clear the canvas before drawing the next histogram

h2.Draw()
c.SaveAs("histogram_y" + "_pmx_only" ".png")




#x = np.linspace(min(predict_x), max(predict_x))
#ax[0].set_xlabel("X predicted position")
#ax[0].set_ylabel("Count")
#ax[0].hist(predict_x)
#ax[1].set_xlabel("Y predicted position")
#ax[1].set_ylabel("Count")
#ax[1].hist(predict_y)
#mu, sigma = norm.fit(predict_x)
#best_fit = norm.pdf(predict_x, mu, sigma)
#plt.plot(predict_x, best_fit)
#plt.show()
#print(all_positions)
#c = ROOT.TCanvas()
#gaus = ROOT.TF1("func", "gaus(0)")
#g.Fit(gaus)
#g.Draw()
#print()
#print("Sigma: ", gaus.GetParameter(2), "µm")
#input()
#print(all_positions)


