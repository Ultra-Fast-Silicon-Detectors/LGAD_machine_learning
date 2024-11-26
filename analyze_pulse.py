import math
import ROOT
ROOT.gROOT.ProcessLine( "gErrorIgnoreLevel = 3005;")

class pulse_analyzer:
    def __init__(self, h, zero_time = 0., correction = 1., invert = False, time_shift = 0., max_points = 1e6):
        self.pulse = h
        self.zero_time = zero_time
        self.correction = correction
        self.invert = invert
        self.time_shift = time_shift
        self.max_points = max_points

    def ADC_to_mV(self):
        for i in range(0, self.pulse.GetNbinsX() + 1):
            self.pulse.SetBinContent(i, self.pulse.GetBinContent(i)*0.2441)

    def correct_pulse(self):
        base = self.get_baseline()
        step = self.pulse.GetBinWidth(0)
        bin_shift = int(self.time_shift/step)
        if step == 0.: step = 1.
        for i in range(0, self.pulse.GetNbinsX() + 1):
            if self.invert: self.pulse.SetBinContent(i - bin_shift, -(self.pulse.GetBinContent(i)-base)*self.correction)
            else: self.pulse.SetBinContent(i - bin_shift, (self.pulse.GetBinContent(i)-base)*self.correction)
            if i > self.max_points: self.pulse.SetBinContent(i - bin_shift, 0.)

    def normalize(self):
        base = self.get_baseline()
        Pmax = self.get_max()
        for i in range(0, self.pulse.GetNbinsX() + 1):
            self.pulse.SetBinContent(i, self.pulse.GetBinContent(i)/Pmax)

    def Draw(self, opt = ""):
        self.pulse.Draw(opt)

    def get_RMS(self):
        val = 0
        for i in range(1, 100):
            val = val + self.pulse.GetBinContent(i)*self.pulse.GetBinContent(i)
        val = math.sqrt(val/100.)
        return val

    def get_jitter(self, rms = 2.5):
        rise = 1000.*self.get_rise_time()
        pmax = self.get_max()
        val = rise/(pmax/rms)
        return val

    def get_baseline(self):
        val = 0
        for i in range(1, 100):
            val = val + self.pulse.GetBinContent(i)
        val = val/100.
        return val

    def get_max(self):
        val = self.pulse.GetBinContent(self.pulse.GetMaximumBin())
        return val

    def get_min(self):
        val = self.pulse.GetBinContent(self.pulse.GetMinimumBin())
        return val

    def get_tmax(self):
        val = self.pulse.GetBinLowEdge(self.pulse.GetMaximumBin())
        return val

    def get_tmin(self):
        val = self.pulse.GetBinLowEdge(self.pulse.GetMinimumBin())
        return val

    def get_area(self):
        val = 0.
        for i in range(0, self.pulse.GetNbinsX()):
            val = val + self.pulse.GetBinContent(i)
        return val

    def get_rise_time(self):
        pmax = self.get_max()
        base = self.get_baseline()

        low = 0.
        high = 0.
        for i in range(self.pulse.GetMaximumBin() - 200, self.pulse.GetMaximumBin()+1):
            if low == 0. and self.pulse.GetBinContent(i) > 0.1*pmax:
                #low = self.pulse.GetBinLowEdge(i)
                if i == self.pulse.GetMaximumBin(): flow = ROOT.TF1("flow", "pol1", self.pulse.GetBinLowEdge(i-2), self.pulse.GetBinLowEdge(i))
                else: flow = ROOT.TF1("flow", "pol1", self.pulse.GetBinLowEdge(i-1), self.pulse.GetBinLowEdge(i+1))
                self.pulse.Fit(flow, "0NQR")
                low = flow.GetX(0.1*pmax)
                #print "low", low
            if high == 0. and self.pulse.GetBinContent(i) > 0.9*pmax:
                #high = self.pulse.GetBinLowEdge(i)
                fhigh = ROOT.TF1("fhigh", "pol1", self.pulse.GetBinLowEdge(i-1), self.pulse.GetBinLowEdge(i+1))
                self.pulse.Fit(fhigh, "0NQR")
                high = fhigh.GetX(0.9*pmax)
                #print "high", high
        val = high - low
        if val < 0.: val = 0.
        if math.isnan(val): val = 0.
        return val

    def get_fall_time(self):
        pmax = self.get_max()
        base = self.get_baseline()

        low = 0.
        high = 0.
        for i in range(self.pulse.GetMaximumBin(), self.pulse.GetNbinsX()):
            if low == 0. and self.pulse.GetBinContent(i) < 0.9*pmax:
                #low = self.pulse.GetBinLowEdge(i)
                flow = ROOT.TF1("flow", "pol1", self.pulse.GetBinLowEdge(i-1), self.pulse.GetBinLowEdge(i+3))
                self.pulse.Fit(flow, "0NQR")
                low = flow.GetX(0.9*pmax)
            if high == 0. and self.pulse.GetBinContent(i) < 0.1*pmax:
                #high = self.pulse.GetBinLowEdge(i)
                fhigh = ROOT.TF1("fhigh", "pol1", self.pulse.GetBinLowEdge(i-2), self.pulse.GetBinLowEdge(i+1))
                self.pulse.Fit(fhigh, "0NQR")
                high = fhigh.GetX(0.1*pmax)
        val = high - low
        #print "IN", low, high, val
        if val < 0.: val = 0.
        if math.isnan(val): val = 0.
        return val

    def get_fwhm(self):
        pmax = self.get_max()
        base = self.get_baseline()

        low = 0.
        high = 0.
        for i in range(self.pulse.GetMaximumBin() - 200, self.pulse.GetMaximumBin()):
            if low == 0. and self.pulse.GetBinContent(i) > 0.5*pmax:
                #low = self.pulse.GetBinLowEdge(i)
                flow = ROOT.TF1("flow", "pol1", self.pulse.GetBinLowEdge(i-2), self.pulse.GetBinLowEdge(i+2))
                self.pulse.Fit(flow, "0NQR")
                low = flow.GetX(0.5*pmax)
        for i in range(self.pulse.GetMaximumBin(), self.pulse.GetNbinsX()):
            if high == 0. and self.pulse.GetBinContent(i) < 0.5*pmax:
                #high = self.pulse.GetBinLowEdge(i)
                fhigh = ROOT.TF1("fhigh", "pol1", self.pulse.GetBinLowEdge(i-2), self.pulse.GetBinLowEdge(i+2))
                self.pulse.Fit(fhigh, "0NQR")
                high = fhigh.GetX(0.5*pmax)
        val = high - low
        if val < 0.: val = 0.
        if math.isnan(val): val = 0.
        return val

    def get_CFD50(self):
        pmax = self.pulsemax()
        base = self.get_baseline()

        time = 0.
        for i in range(self.pulse.GetMaximumBin() - 10, self.pulse.GetMaximumBin()):
            if time == 0. and self.pulse.GetBinContent(i) > 0.5*pmax:
                #time = self.pulse.GetBinLowEdge(i)
                fcfd = ROOT.TF1("fcfd", "pol1", self.pulse.GetBinLowEdge(i-2), self.pulse.GetBinLowEdge(i+1))
                self.pulse.Fit(fcfd, "0NQR")
                time = fcfd.GetX(0.5*pmax)
                # if CFD50 is plainly wrong
                #if (time > self.get_tmax()) or (time < (self.get_tmax() - 1.)): time = self.pulse.GetBinLowEdge(i)
                if math.isnan(time): time = 0. #self.pulse.GetBinLowEdge(i)
        return time

    def pulsemax(self):
        pmxtime = self.pulse.GetMaximumBin()
        gaussian = ROOT.TF1("Fit", "gaus", self.pulse.GetBinLowEdge(pmxtime-2), self.pulse.GetBinLowEdge(pmxtime+2))
        self.pulse.Fit(gaussian, "ONQR")
        time = gaussian.GetParameter(0)
        return time


    def get_CFD30(self):
        pmax = self.get_max()
        base = self.get_baseline()

        time = 0.
        for i in range(self.pulse.GetMaximumBin() - 200, self.pulse.GetMaximumBin()):
            if time == 0. and self.pulse.GetBinContent(i) > 0.3*pmax:
                #time = self.pulse.GetBinLowEdge(i)
                fcfd = ROOT.TF1("fcfd", "pol1", self.pulse.GetBinLowEdge(i-2), self.pulse.GetBinLowEdge(i+2))
                self.pulse.Fit(fcfd, "0NQR")
                time = fcfd.GetX(0.3*pmax)
                # if CFD30 is plainly wrong
                #if time > self.get_tmax() or time < (self.get_tmax() - 1): time = self.pulse.GetBinLowEdge(i)
                if math.isnan(time): time = self.pulse.GetBinLowEdge(i)
        return time

    def get_time_threshold(self, threshold):
        pmax = self.get_max()
        base = self.get_baseline()

        time = 0.
        for i in range(self.pulse.GetMaximumBin() - 200, self.pulse.GetMaximumBin()):
            if time == 0. and self.pulse.GetBinContent(i) > threshold:
                #time = self.pulse.GetBinLowEdge(i)
                fcfd = ROOT.TF1("fcfd", "pol1", self.pulse.GetBinLowEdge(i-2), self.pulse.GetBinLowEdge(i+2))
                self.pulse.Fit(fcfd, "0NQR")
                time = fcfd.GetX(threshold)
        if math.isnan(time): val = 0.
        return time
