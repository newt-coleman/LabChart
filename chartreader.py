import pandas as pd    
from scipy.io import loadmat
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt

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
    
    def detect_spike(self, data, threshold, kind=None):
        """
        Identifies action potentials in a recording based on a threshold detection
        
        param: data       sequence of recording
        param: threshold  int or float of the cutoff
        param: kind       string parameter that dictates what other features associated
                          with the spikes to return.
                            - 'amp' returns list of max-min for each ap
                            - 'max' return list of max
                            - 'course' returns a list of the recording course of an ap
                          
        return:           time course of the recording that identifies when an ap has occured
        """
        features = []
        times = data*0
        for t in range(len(data)-2):
            temp_start = 0
            
            if data[t+1] > data[t] and data[t+1] > data[t+2] and data[t+1] > threshold:
                ap_time = t + 1
                times[ap_time] = 1
                ap_course = data[ap_time - self._samples(0.0015) : ap_time + self._samples(0.003)]
                if kind == 'amp':
                    features.append(max(ap_course) - min(ap_course))
                elif kind == 'max':
                    features.append(max(ap_course))
                elif kind == 'course':
                    features.append(ap_course)
            
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

