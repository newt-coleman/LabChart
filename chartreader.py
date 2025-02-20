import pandas as pd
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

#### Todo ####
# functionalize everything
# create averages over subpages
# work in comments
# implement spike detector
file = loadmat('Lab3 - worm stim/Lab_3.mat') # 'datastart', end, samplerate has shape (x, y) where x is the
                                             # number of channels (names given by 'titles') and y is number of trial or scope
                                             # pages.

pages = [1, 1, 1, 1, 1, 1, 1]
channels = file['titles']
data = np.array(file['data'])

time_index = []
time_course = []
for page in range(len(pages)):
    for subpage in range(pages[page]):
        page_num = int(np.sum(pages[0:page]) + subpage)
        time_index.append((page, subpage))
        samples = file['dataend'][0, page_num] - file['datastart'][0, page_num] + 1
        time_course.append(np.linspace(0, samples / file['samplerate'][0, page_num], int(samples)))
        
time_index = pd.MultiIndex.from_tuples(time_index, names=['Page', 'Subpage'])
time_course = pd.Series(time_course, index=time_index)


scope_data = []
scope_index = []
comments = []
file_com = file['com']
for channel in range(len(channels)):
    for page in range(len(pages)):
        for subpage in range(pages[page]):
            net_page = int(np.sum(pages[0:page]) + subpage)
            
            start = int(file['datastart'][channel, net_page] - 1)
            end = int(file['dataend'][channel, net_page])
            scope_data.append(data[0, start:end])
            
            scope_index.append((channels[channel], page, subpage))
            page_coms = []
            for i in range(len(file_com[:, 0])):
                if file_com[i, 0]-1 == channel and file_com[i, 1]-1 == net_page:
                    page_coms.append(time_course[(page, subpage)][int(file_com[i, 2])])
            comments.append(page_coms)
                    
            
scope_index = pd.MultiIndex.from_tuples(scope_index, names=['Channel', 'Page', 'Trial'])
scope_data = pd.Series(scope_data, index=scope_index)
comments = pd.Series(comments, index=scope_index)

fig, (chan1, chan2) = plt.subplots(2, 1)
chan1.vlines(comments[('Channel 1', 5, 0)], 0, 1, transform=chan1.get_xaxis_transform(), colors='r')
chan1.plot(time_course[(5, 0)], scope_data[('Channel 1', 5, 0)])
chan2.plot(time_course[(5, 0)], scope_data[('Channel 2', 5, 0)])
plt.show()

#print(file['dataend'][0,0])
print(data.shape)

#print(file['samplerate'].shape)
