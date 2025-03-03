import pandas as pd
from scipy.io import loadmat
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
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
    
    def _samples(self, seconds):
        return int(seconds * 40000)
    
    def detect_spike(self, data, threshold, kind=None, before=0.0015, after=0.003):
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
        features = []
        times = data*0
        for t in range(len(data)-2):
            temp_start = 0
            
            if ((data[t+1] > data[t] and data[t+1] > data[t+2]) or (data[t+1] == data[t] and data[t+1] > data[t+2])) and data[t+1] > threshold:
                ap_time = t + 1
                times[ap_time] = 1
                start = ap_time - self._samples(before)
                end = ap_time + self._samples(after)
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
            
        if kind != None:
            return times, features
        return times
    
    def avg_subpages(self, page_num):
        avg =[]
        for channel in self.channels:
            pages = self.scope_data[(channel, page, None)].to_numpy()
            avg.append(np.avg(pages, axis=0))
        return avg
                
#             
#         fig, channels = plt.subplots(channel_num, 1)
#         for channel in channels:
#             channel
#         return data
# 
# def detect_spike(data, threshold):
#     time_course = data[0]
#     scope_data = data[1]
#     
#     events = time_course.copy()

# def avg_subpages(data):
#     for channel in data[0]()


##########WACKY TOWN YOURE DOING TOO MUCH ABOVE####################
###################################################################
######################Lab 3 specifics below########################
amp = [1000, 1, 1]
demo = LabChart('Lab_3_Demo.mat', amp, structure=[5, 5, 30])
dtime, dscope = demo.time_course, demo.scope_data

data = LabChart('Lab_3', amp)
time_course, scope_data= data.time_course, data.scope_data

def samples(seconds):
    return int(seconds * 40000)

####################FIGURE 1#########################
#### Step 1: sample evoked and non-evoked recording
## use bandpass filter (30, 1000)
evoked_t = time_course[(6, 0)][samples(6):samples(16)]
evoked = data.bpass(('Channel 1', 6, 0), (30, 1000), 40000)[samples(6):samples(16)]
spont_t = time_course[(1, 0)][samples(5):samples(15)]
spont = data.bpass(('Channel 1', 1, 0), (30, 1000), 40000)[samples(5):samples(15)]

#### Step 2: spike detection with threshold at 1 mV
evoked_times, evoked_amps = data.detect_spike(evoked, 1.0, kind='amp')
spont_times, spont_amps = data.detect_spike(spont, 1.0, kind='amp')
# fig, (ax1, ax2) = plt.subplots(2,1)    #### Visualizing spike ids
# ax1.plot(evoked_t, evoked, lw=0.2)
# ax2.plot(evoked_t, evoked_times)
# plt.show()

#### Step 3: plot two histograms
plt.rcParams['font.size'] = 14
fig, (axstim, axspont) = plt.subplots(1,2)
bins = np.arange(0, 66, 5)
axspont.hist(spont_amps, log=True, bins=bins, color='#282828')
axstim.hist(evoked_amps, log=True, bins=bins, color='#282828')
axstim.set_ylim(0.9,335)
axspont.set_ylim(0.9,335)
axspont.set_xlabel("Spontaneous activity")
axstim.set_xlabel("Stimulated activity")
axstim.set_ylabel('Number of action potentials')
fig.supxlabel("Action potential amplitude (mV)")
plt.show()

#### Step 4: plotting raw data
# fig, (axstim, axspont) = plt.subplots(2, 1)
# axstim.plot(evoked_t, evoked, color='#282828', lw=0.3)
# axstim.set_ylim(-28, 38)
# axstim.set_xticks([])
# axstim.set_yticks([])
# axstim.spines.bottom.set_bounds(5.75, 6.75)
# # axstim.spines.bottom.set_position(('data', -0.5))
# axstim.spines.left.set_bounds(-28, -13)
# axstim.spines.left.set_position(('data', 5.75))
# # Hide the right and top spines
# axstim.spines.right.set_visible(False)
# axstim.spines.top.set_visible(False)
# axspont.annotate('1 s', (5.05, 17))
# axspont.annotate('15 mV', (4.5, 23.25), rotation=90)
# 
# axspont.plot(spont_t, spont, color='#282828', lw=0.3)
# axspont.set_ylim(-28, 38)
# axspont.set_axis_off()
# plt.tight_layout(h_pad=-5)
# plt.show()

#### Step 5: other numbers might be useful
# print("Spontaneous spike frequency: " + str(len(spont_amps) / 10))
# print("Evoked spike frequency: " + str(len(evoked_amps) / 10))
# evoked_times, e_ap = data.detect_spike(evoked, 1.0, kind='course', before=0.002)
# spont_times, s_ap = data.detect_spike(evoked, 1.0, kind='course', before=0.002)
# e_ap.extend(s_ap)
# aps = np.array((e_ap))
# 
# high_amp=[]
# for ap in aps:
#     if np.max(ap) > 5:
#         high_amp.append(ap)
# high_amp = np.array((high_amp))
# 
# plt.rcParams['font.size'] = 14
# fig, (ax1, ax2) = plt.subplots(1,2)
# avg_ap = np.average(high_amp, axis=0)
# print(high_amp.shape)
# ax1.plot(np.linspace(0, len(avg_ap)/40, len(avg_ap)), avg_ap, color='#282828')
# ax1.set_xlabel('time (ms)')
# ax1.set_ylabel('amplitude (mV)')
# 
# low_amp=[]
# for ap in aps:
#     if np.max(ap) < 5:
#         low_amp.append(ap)
# low_amp = np.array((low_amp))
# 
# avg_ap = np.average(low_amp, axis=0)
# print(low_amp.shape)
# ax2.plot(np.linspace(0, len(avg_ap)/40, len(avg_ap)), avg_ap, color='#282828')
# ax2.set_xlabel('time (ms)')
# plt.show()


####################FIGURE 2#########################
#### Step 1: Generate probabilities

# stimulus = np.zeros((30,31))
# ap = np.zeros((30,31))
# 
# for page in range(30):
#     for stim in range(31):
#         minV = np.min(dscope[('Channel 1', 2, page)][samples(stim + 0.005) : samples(stim + 0.995)])
#         if minV <= -200:
#             ap[page, stim] = 1
#         stimulus[page, stim] = np.max(dscope[('Channel 2', 2, page)][samples(stim) : samples(stim + 0.05)])
#         
# stim = np.round(np.average(stimulus, axis = 0), decimals=1) ### Consolidate same stimulus vals
# stimulus = np.unique(stim)                                  ### I am also POSITIVE there is a more elegant way
# ap = np.average(ap, axis = 0)                               ### because this feels kinda janky
# ap_prob = []
# k=0
# while k < len(ap):
#     i = 1
#     while k + i < len(ap) and stim[k] == stim[k+i]:
#         i += 1
#     ap_prob.append(np.mean(ap[k:k+i]))
#     k=k+i
# 
# ap_prob = np.array(ap_prob)




# #### Step 2: Fit the sigmoid

def sumSquaredError(a,b,c):
    y = lambda t: a + (1 / (1 + np.exp((t-b)/-c)))    # Define the sigmoid model
    error = sum(np.abs(y(stimulus) - ap_prob)**2)     # Compute the error using sum-of-squared error
    return error

def logistic(p, t):
    return (1 / (1 + np.exp((t-p[1])/-p[2])))

def r_sq(y, yhat):
    return 1 - (np.sum((y-yhat)**2) / np.sum((y-np.mean(y))**2))
    
# 
# adapter = lambda p: sumSquaredError(p[0], p[1], p[2])
# guess = np.array([0, 1, 1])
# fit = scipy.optimize.fmin(adapter, guess)
# print(fit)
# print("Rsquare = " + str(r_sq(ap_prob, logistic(fit, stimulus))))

# #### Step 3: Plot fitted function
# plt.rcParams['font.size'] = 14
# plt.plot(stimulus, ap_prob, 'o', color='#282828')
# smoothx = np.linspace(1.5, 3.5, 100)
# plt.plot(smoothx, logistic(fit, smoothx), 'r', label="$\mu$ = 2.79")
# plt.xlabel("Stimulus amplitude (V)")
# plt.ylabel("Probability of action potential")
# plt.xticks([1.5, 2, 2.5, 3, 3.5])
# plt.legend()
# plt.show()

#### Step 4: plot raw data
# demo_data = demo.bpass(('Channel 1', 2, 0), (30, 1000), 40000)
# demo_data = dscope[('Channel 1', 2, 0)]
# neg_demo = demo_data *-1
# plt.rcParams['font.size'] = 12
# fig, (chan1, chan2) = plt.subplots(2, 1)
# chan1.plot(dtime[(2, 0)], neg_demo, color='#282828', lw=0.4)
# chan1.set_xticks([])
# chan1.set_yticks([])
# chan1.spines.bottom.set_bounds(-0.5, 4.5)
# chan1.spines.bottom.set_position(('data', -600))
# chan1.spines.left.set_bounds(-600, -300)
# chan1.spines.left.set_position(('data', -0.5))
# # # Hide the right and top spines
# chan1.spines.right.set_visible(False)
# chan1.spines.top.set_visible(False)
# 
# chan2.plot(dtime[(2,0)], dscope[("Channel 2", 2, 0)], color='#282828', lw=0.4)
# chan2.set_axis_off()
# chan2.annotate('5 s', (1.2, 3.15))
# chan2.annotate('300 mV (top) / 1 V (bottom)', (-1.6, 2.8), rotation=90)
# plt.tight_layout(h_pad=-0.2)
# plt.show()

#### Step 5: Miscellaneous
# spike_times = []
# for k in range(30):
#     neg_demo = -1 * dscope[('Channel 1', 2, k)]
#     spike_times.extend(demo.detect_spike(neg_demo, 200, kind='max_times')[1])
# diffs = []
# time_proxy = []
# prev = spike_times[0]
# for k in range(len(spike_times) - 1):
#     if spike_times[k + 1][0] < 500 and prev[0] > 300:
#         diffs.append((spike_times[k + 1][1] - prev[1])/40)
#         print(diffs[-1])
#         time_proxy.append(prev[1]/40000)
#     prev = spike_times[k+1]
# hist, bin_edges = np.histogram(diffs)
# print(bin_edges)
# print(np.mean(diffs))
# print(np.std(diffs))
# plt.rcParams['font.size'] = 14
# 
# plt.hist(diffs, color='#282828')
# plt.xlabel('Conduction time (ms)')
# plt.ylabel('Count')
# plt.show()



#### Step 6: zoom in on successful/ unsuccessful trial
# demo_data = demo.bpass(('Channel 1', 2, 0), (30, 1000), 40000)
# fail = dscope[('Channel 1', 2, 0)][0:20000]
# success = dscope[('Channel 1', 2, 0)][1160000:1180000]
# plt.rcParams['font.size'] = 12
# fig, (chan1, chan2) = plt.subplots(1, 2)
# chan1.plot(dtime[(2, 0)][0:20000], fail, color='#282828', lw=0.4)
# chan1.set_xticks([])
# chan1.set_yticks([])
# chan1.spines.bottom.set_bounds(0.4, 0.5)
# chan1.spines.bottom.set_position(('data', -500))
# chan1.spines.left.set_bounds(-500, -300)
# chan1.spines.left.set_position(('data', 0.4))
# # # Hide the right and top spines
# chan1.spines.right.set_visible(False)
# chan1.spines.top.set_visible(False)
# 
# chan2.plot(dtime[(2,0)][1160000:1180000], success, color='#282828', lw=0.4)
# chan2.set_axis_off()
# chan1.annotate('100 ms', (0.4, -545))
# chan1.annotate('200 mV', (0.37, -490), rotation=90)
# plt.tight_layout(h_pad=-0.2)
# plt.show()
