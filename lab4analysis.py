import pandas as pd    
from scipy.io import loadmat
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, find_peaks
from scipy import stats

#### Todo ####
### functionalize everything
# create averages over subpages
### work in comments 
### implement spike detector
class LabChart:
    """
    fields: channels, long form data, timecourse, page form data, index for said, comments
    """
    def __init__(self, path, factor, structure=None):
        """
        Loads and structures a LabChart file into page and subpages.
        
        param: path       string of the path to the '.mat' file
        param: factor     array of the factor to amplify each channel by
        param: structure  int array of the page structure. Each value represents the number of subpages a
                          page has. The length of the array represents how many pages there are 
                          
        return:           tuple of Series representing the time course information for each page/subpage,
                          Series representing the recorded data for each channel/page/subpage, as well as
                          a Series representing the commenting info for each channel/page/subpage
                          
        """
        file = loadmat(path)               # 'datastart', end, samplerate has shape (x, y) where x is the                                  
        self.channels = file['titles']     # number of channels (names given by 'titles') and y is number of trial or scope
        self.data = np.array(file['data']) # pages.
        if structure == None:
            structure = np.ones(len(file['datastart'][0, :]), dtype=int)
        self.pages=structure
        # TODO: check that structure actually matches the number of datastarts
        time_index = []               # initialize time course first
        time_course = []              # only multi-indexed (page, subpage)
        for page in range(len(self.pages)):
            for subpage in range(self.pages[page]):
                page_num = int(np.sum(self.pages[0:page]) + subpage)
                time_index.append((page, subpage))
                samples = file['dataend'][0, page_num] - file['datastart'][0, page_num] + 1
                time_course.append(np.linspace(0, samples / file['samplerate'][0, page_num], int(samples)))
                
        self.time_index = pd.MultiIndex.from_tuples(time_index, names=['Page', 'Subpage'])
        self.time_course = pd.Series(time_course, index=time_index)


        scope_data = []               # initialize recorded data and comment info
        scope_index = []              # two separate series that use the same indexing
        comments = []
        file_com = file['com']
        for channel in range(len(self.channels)):
            for page in range(len(self.pages)):
                for subpage in range(self.pages[page]):
                    net_page = int(np.sum(self.pages[0:page]) + subpage)
                    
                    start = int(file['datastart'][channel, net_page] - 1)
                    end = int(file['dataend'][channel, net_page])
                    scope_data.append(self.data[0, start:end]*factor[channel])
                    
                    scope_index.append((self.channels[channel], page, subpage)) # multi-indexed (channel, page, subpage)
                    page_coms = []
                    for i in range(len(file_com[:, 0])):
                        if file_com[i, 0]-1 == channel and file_com[i, 1]-1 == net_page:
                            page_coms.append(self.time_course[(page, subpage)][int(file_com[i, 2])])
                    comments.append(page_coms)
                            
                    
        self.scope_index = pd.MultiIndex.from_tuples(scope_index, names=['Channel', 'Page', 'Trial'])
        self.scope_data = pd.Series(scope_data, index=scope_index)
        self.comments = pd.Series(comments, index=scope_index)
        

    def scope_vis(self, channel_num):
        avg = []
        for page in range(len(self.pages)):
            avg.append(np.average(np.array([self.scope_data(channel_num, page, None)])))
        return avg
        
    def hpass(self, index, cutoff, sampling):
        sos = butter(5, cutoff, btype='highpass', fs=sampling, output='sos')
        passed = sosfiltfilt(sos, self.scope_data[index])
        return passed
    
    def bpass(self, index, cutoff, sampling):
        sos = butter(5, cutoff, btype='bandpass', fs=sampling, output='sos')
        passed = sosfiltfilt(sos, self.scope_data[index])
        return passed
    
    def detect_spike(self, data, threshold, kind=None, before=0.0015, after=0.003, fs=40000):
        """
        Identifies action potentials in a recording based on a threshold detection
        
        param: data       sequence of recording
        param: threshold  int or float of the cutoff
        param: kind       string parameter that dictates what other features associated
                          with the spikes to return.
                            - 'amp' returns list of max-min for each ap
                            - 'max' return list of max
                            - 'course' returns a list of the recording course of an ap
                            - 'max_times' returns a list of tuples of (max_amp, index)
                          
        return:           time course of the recording that identifies when an ap has occured
        """
        data = np.array(data)
        features = []
        times = data*0
        before = int(before * fs)
        after = int(after * fs)
        t = 0
        while t < len(data)-2:          
            
            if ((data[t+1] > data[t] and data[t+1] > data[t+2]) or (data[t+1] == data[t] and data[t+1] > data[t+2])) and data[t+1] > threshold:
                ap_time = t + 1
                times[ap_time] = 1
                start = ap_time - before
                end = ap_time + after
                if start < 0:
                    start = 0
                if end > len(data):
                    end = -1
                ap_course = data[start : end]
                if kind == 'amp':
                    features.append(max(ap_course) - min(ap_course))
                elif kind == 'max':
                    features.append(max(ap_course))
                elif kind == 'course':
                    features.append(ap_course)
                elif kind == 'max_times':
                    features.append((max(ap_course), t+1))
                t += after
            t += 1
            
        if kind != None:
            return times, features
        return times
    
    def avg_subpages(self, page_num):
        avg =[]
        for channel in self.channels:
            pages = []
            for subpage in range(self.pages[page_num]):
                pages.append(self.scope_data[(channel, page_num, subpage)])
            avg.append(np.average(pages, axis=0))
        return np.array(avg)
    
    def save_pages(self, path):
        for page in self.scope_index:
            axs = np.array(len(self.channels))
            fig, axs = plt.subplots(len(self.channels), 1)
            for k in range(len(axs)):
                axs[k].plot(self.time_course[page[1:3]], self.scope_data[(self.channels[k], page[1], page[2])])
            plt.savefig(str(path) + "/Page" + str(page[1]) + "subpage" + str(page[2]) + ".png")
            plt.close()
                
demo = LabChart("Lab_4_Demo_Data.mat", [0.00624, 0.00312], structure=[5, 5, 5, 5, 5])
## input is in nanoamps, recording is in milivolts

####### FIGURE 1: time constants + subthreshold activity ######
taus = []
n1 = demo.avg_subpages(0) ## first 8 hyperpols are good from 0.5s-8.0s sampling rate is 100kH
n2 = demo.avg_subpages(2) ## first 8 depols are good from 3.5s on
n3 = demo.avg_subpages(4) ## first 4 hyperpols are good

def calc_tau(scope, trials, sampling):
    wform = scope[0]
    stimform = scope[1]
    
    stims = []
    deflections = []
    tau = []
    for trial in range(trials):
        t = trial*sampling
        deflections.append(min(wform[t:int((trial*sampling)+(0.5*sampling))]) - wform[t])
        stims.append(min(stimform[t:int((trial*sampling)+(0.5*sampling))]) - stimform[t])
        characteristic_amp = ((min(wform[t:int((trial*sampling)+(0.5*sampling))]) - wform[t]) * 0.63) + wform[t]
        while wform[t] > characteristic_amp:
            t += 1      
        tau.append((t - trial*sampling)/sampling)
    return np.array(tau), np.array(stims), np.array(deflections)

def calc_rad(conductance, tau):
    SPECIFIC_CAPACITANCE = 10**-8 # farads per sq mm i think
    cond = conductance * 10**-6 # puts conductance in siemens
    return np.sqrt((cond * tau)/(np.pi * SPECIFIC_CAPACITANCE)) # returns in milimeters

tau1, stims1, v1 = calc_tau(n1[:, 50000:], 8, 100000)
tau2, stims2, v2 = calc_tau(-1 * n2[:, 140000:], 8, 40000)
tau3, stims3, v3 = calc_tau(n3[:, 20000:], 4, 40000)


cond1 = stats.linregress(v1, stims1)
cond2 = stats.linregress(v2, stims2)
cond3 = stats.linregress(v3, stims3)
print(calc_rad(cond1.slope, np.mean(tau1)))
print(calc_rad(cond2.slope, np.mean(tau2)))
print(calc_rad(cond3.slope, np.mean(tau3)))
print(cond1.rvalue**2)
print(cond2.rvalue**2)
print(cond3.rvalue**2)
print("Neuron 1 conductance: " + str(cond1.slope)) #microsiemens
print("Neuron 2 conductance: " + str(cond2.slope))
print("Neuron 3 conductance: " + str(cond3.slope))
print("Neuron 1 tau: " + str(np.mean(tau1)) + "\nSTD: " + str(np.std(tau1))) #in seconds
print("Neuron 2 tau: " + str(np.mean(tau2)) + "\nSTD: " + str(np.std(tau2)))
print("Neuron 3 tau: " + str(np.mean(tau3)) + "\nSTD: " + str(np.std(tau3)))

#### Plots barchart comparing taus
# plt.rcParams['font.size'] = 15
# plt.boxplot([tau1*1000, tau2*1000, tau3*1000], tick_labels = ['Cell 1', 'Cell 2', 'Cell 3'])
# plt.ylabel(r'$\tau$ (ms)')
# plt.tight_layout()
# plt.show()

#### Plot IV curve
# span = np.linspace(-65, -1.8)
# plt.plot(v1, stims1, 'o', label='Cell 1', color='tab:blue')
# plt.plot(span, cond1.slope*span + cond1.intercept, color='tab:blue')
# plt.plot(v2, stims2, 'o', label='Cell 2', color='tab:orange')
# plt.plot(span, cond2.slope*span + cond2.intercept, color='tab:orange')
# plt.plot(v3, stims3, 'o', label='Cell 3', color='tab:green')
# plt.plot(span, cond3.slope*span + cond3.intercept, color='tab:green')
# plt.xlabel("Voltage deflections (mV)")
# plt.ylabel("Input current (nA)")
# plt.legend()
# plt.tight_layout()
# plt.show()

#### Plot raw subthreshold trace || neuron 1 averaged for first 8.5s
# END_SAMPLE = 850000
# fig, (chan1, chan2) = plt.subplots(2, 1)
# chan1.plot(demo.time_course[(0, 0)][:END_SAMPLE], n1[0, :END_SAMPLE], color='#282828', lw=0.4)
# chan1.set_xticks([])
# chan1.set_yticks([])
# chan1.spines.bottom.set_bounds(7.5, 8.5)
# chan1.spines.bottom.set_position(('data', -57))
# chan1.spines.right.set_bounds(-52, -57)
# chan1.spines.right.set_position(('data', 8.5))
# chan1.annotate('1 s', (7.75, -59))
# chan1.annotate('5 mV/ 0.2 nA', (8.55, -61.5), rotation=270)
# # # Hide the left and top spines
# chan1.spines.left.set_visible(False)
# chan1.spines.top.set_visible(False)
# 
# chan2.plot(demo.time_course[(0, 0)][:END_SAMPLE], n1[1, :END_SAMPLE], color='#282828', lw=0.4)
# chan2.set_axis_off()
# plt.tight_layout(h_pad=-0.2)
# plt.show()

####### FIGURE 2: threshold and adaptation
#### Identify threshold - manual
plt.rcParams['font.size'] = 15
Es = np.array([np.mean(n1[0, :50000]), np.mean(n2[0, :20000]), np.mean(n3[0, :20000])])
print(Es)
thresh1 = np.array([0.25, 0.35, 0.25, 0.35, 0.35])
thresh2 = np.array([5.65, 5.65, 7.45, 6.275, 5.65]) 
thresh3 = np.array([0.7, 0.7, 0.7, 0.7, 0.7])

thresh = np.array([thresh1/cond1.slope + Es[0], thresh2/cond2.slope + Es[1], thresh3/cond3.slope + Es[2]])
print("THreshold: " + str(np.mean(thresh, axis=1)))
cells = [1, 2, 3]
plt.errorbar(cells, np.mean(thresh, axis=1), yerr=np.std(thresh, axis=1),
             linestyle='none', marker='o', label='Threshold')
plt.plot(cells, Es, marker='s', linestyle='none', label='Rest')
plt.xticks(cells, labels=['Cell 1', 'Cell 2', 'Cell 3'])
plt.xlim(0.5,3.5)
plt.ylabel("Membrane potential (mV)")
plt.legend()
plt.tight_layout()
plt.show()

#### trace of threshold spiking
# START = 800000
# END = 1150000
# sample = demo.scope_data[('Channel 1', 0, 3)][START:END]
# current = demo.scope_data[('Channel 2', 0, 3)][START:END]
# time = demo.time_course[(0, 0)][START:END]
# fig, (chan1, chan2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]})
# chan1.plot(time, sample, color='#282828', lw=0.4)
# x = np.linspace(8, 11.5, 3)
# chan1.plot(x, np.ones(3)*np.mean(thresh, axis=1)[0], 'r', linestyle='dashed')
# chan1.set_xlim(8, 11.5)
# chan1.set_xticks([])
# chan1.set_yticks([])
# chan1.spines.bottom.set_bounds(8.25, 8.75)
# chan1.spines.bottom.set_position(('data', -20))
# chan1.spines.left.set_bounds(-20, 0)
# chan1.spines.left.set_position(('data', 8.25))
# chan1.annotate('1 s', (8.38, -24))
# chan1.annotate('20 mV/ 0.3 nA', (8.1, -25), rotation=90)
# # # # Hide the left and top spines
# chan1.spines.right.set_visible(False)
# chan1.spines.top.set_visible(False)
# 
# 
# chan2.plot(time, current, color='#282828', lw=0.4)
# chan2.set_axis_off()
# chan2.set_xlim(8, 11.5)
# plt.tight_layout(h_pad=-0.2)
# plt.show()

####### adaptation
freqs1 = []
freqs2 = []
widths = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
for k in range(5):
    train = demo.scope_data[('Channel 1', 1, k)]
    times, features = find_peaks(train, height=0, width=500) # for n1, h=0, width = 500 works well, n2 h=150, width = 400
    hist, edges = np.histogram(times/100000, bins=widths)
    freqs1.append(hist)
    
for k in range(5):
    train = demo.scope_data[('Channel 1', 3, k)]
    times, features = find_peaks(train, height=100, width=400) # for n2, h=0, width = 500 works well, n2 h=150, width = 400
    hist, edges = np.histogram(times/40000, bins=widths)
    freqs2.append(hist)
    
freqs1 = np.array(freqs1)
mean1, std1 = np.average(freqs1, axis=0), np.std(freqs1, axis=0)
freqs2 = np.array(freqs2)
mean2, std2 = np.average(freqs2, axis=0), np.std(freqs2, axis=0)
mean2[4:] = 0  # manual correction for spike detection algorithm
std2[4:] = 0

# plt.errorbar(widths[:-1], mean1, marker='o', yerr=std1, linestyle='none', label='Cell 1')
# plt.errorbar(widths[:-1], mean2, marker='o', yerr=std2, linestyle='none', label='Cell 2')
# plt.xlabel("Time bins (s)")
# plt.ylabel("Spike counts")
# plt.xticks([0, 2, 4, 6, 8, 10], labels=[0.5, 2.5, 4.5, 6.5, 8.5, 10.5])
# plt.legend()
# plt.tight_layout()
# plt.show()

####### adaptation trace
# START = 50000
# END = 1150000
# sample = demo.scope_data[('Channel 1', 1, 4)][START:END]
# time = demo.time_course[(1, 4)][START:END]
# 
# fig, (chan1) = plt.subplots()
# chan1.plot(time, sample, color='#282828', lw=0.4)
# chan1.set_xticks([])
# chan1.set_yticks([])
# chan1.spines.bottom.set_bounds(10.5, 11.5)
# chan1.spines.bottom.set_position(('data', -12))
# chan1.spines.right.set_bounds(-12, 2)
# chan1.spines.right.set_position(('data', 11.5))
# chan1.annotate('1 s', (10.65, -15))
# chan1.annotate('20 mV', (11.5, -10.65), rotation=270)
# # # # Hide the left and top spines
# chan1.spines.left.set_visible(False)
# chan1.spines.top.set_visible(False)
# 
# plt.tight_layout()
# plt.show()


##### individual peak dynamics
## Neuron 1
fig, (ax1, ax2, ax3) = plt.subplots(1,3)
fig.subplots_adjust(wspace=0.1)

aps1 = []
widths1 = []
heights1 = []
for k in range(5):
    train = demo.scope_data[('Channel 1', 0, k)][90000:]
    times, features = find_peaks(train, height=0, width=500)
    widths1.extend(features['widths'])
    for time in times:
        ap = np.array(train[time-10000:time+10000])
        aps1.append(ap)
        heights1.append(np.max(ap) - np.min(ap))
aps1 = np.array(aps1)
avg1 = np.mean(aps1, axis=0)
offset1 = Es[0] - avg1[0]
ax1.plot(np.linspace(0, 200, len(avg1)), avg1 + offset1, color='#282828')
ax1.set_ylim(-63, 12.5)
ax1.set_ylabel("Membrane potential (mV)")


aps2 = []
widths2 = []
heights2 = []
for k in range(5):
    train = demo.scope_data[('Channel 1', 2, k)][90000:]
    times, features = find_peaks(train, height=73.75, width=500)
    widths2.extend(features['widths'])
    for time in times:
        ap = np.array(train[time-2400:time+6000])
        aps2.append(ap)
        heights2.append(np.max(ap) - np.min(ap))
aps2 = np.array(aps2)
avg2 = np.mean(aps2, axis=0)
offset2 = Es[1] - avg2[0]
ax2.plot(np.linspace(0, 210, len(avg2)), avg2 + offset2, color='#282828')
ax2.set_ylim(-63, 12.5)
ax2.set_yticks([])
# plt.show()

aps3 = []
prominences = []
heights3 = []
for k in range(5):
    train = demo.scope_data[('Channel 1', 4, k)][220000:600000]
    for j in range(10):
        threshold = 4*j +1
        times, spikes = demo.detect_spike(train[j*40000: j*40000 + 20000], threshold, kind='course', before=0.01, after=0.06)
        for spike in spikes:
            if len(spike) == 2800:
                heights3.append(np.max(spike) - np.min(spike[np.argmax(spike):]))
                aps3.append(spike)
print(len(aps3))
aps3 = np.array(aps3)
avg3 = np.mean(aps3, axis=0)
offset3 = Es[2] - avg3[0]
ax3.plot(np.linspace(0, 110, len(avg3)), avg3 + offset3, color='#282828')
ax3.set_ylim(-63, 12.5)
ax3.set_yticks([])
fig.supxlabel("Time (ms)")
plt.tight_layout()
plt.show()
plt.hist(heights3)
plt.xlabel("Spike Amplitude (mV)")
plt.ylabel("Counts")
plt.tight_layout()
plt.show()

print("N1 amp: " + str(np.mean(heights1)))
print("N2 amp: " + str(np.mean(heights2)))
print("N3 amp higher mode: " + str(np.mean([height for height in heights3 if height > 10])))
print("N3 amp lower mode: " + str(np.mean([height for height in heights3 if height < 10])))
plt.boxplot([np.array(heights1), np.array(heights2), np.array(heights3)],
            tick_labels=["Cell 1", "Cell 2", "Cell 3"], showfliers=False)
plt.ylabel("Spike Amplitude (mV)")
plt.tight_layout()
plt.show()

####### cell 3 trace
START = 620000
END = 640000
sample = demo.scope_data[('Channel 1', 4, 1)][START:END]
time = demo.time_course[(4, 1)][START:END]

fig, (chan1) = plt.subplots()
chan1.plot(time, sample, color='#282828')
chan1.set_xticks([])
chan1.set_yticks([])
chan1.spines.bottom.set_bounds(15.85, 15.95)
chan1.spines.bottom.set_position(('data', 10))
chan1.spines.right.set_bounds(10, 20)
chan1.spines.right.set_position(('data', 15.95))
chan1.annotate('100 ms', (15.86, 6))
chan1.annotate('10 mV', (15.97, 11), rotation=270)
# # # Hide the left and top spines
chan1.spines.left.set_visible(False)
chan1.spines.top.set_visible(False)

plt.tight_layout()
plt.show()



train = demo.scope_data[('Channel 1', 4, 0)]
train1 = demo.scope_data[('Channel 2', 4, 0)]
# times, features = find_peaks(train, height=170, width=800)
# print(features['widths'])
# fig, (chan1, chan2) = plt.subplots(2, 1)
# chan1.plot(demo.time_course[(3,0)][:480000], train[:480000])
# chan2.vlines(times, -10, 10)
# plt.show()

n1 = train
fig, (chan1, chan2) = plt.subplots(2, 1)
chan1.plot(demo.time_course[(4, 0)], train)
chan2.plot(demo.time_course[(4,0)], train1)
plt.show()


