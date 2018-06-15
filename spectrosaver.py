import os
import wave
import matplotlib
matplotlib.use("tkagg")
from matplotlib import pylab


#import the pyplot and wavfile modules 

import matplotlib.pyplot as plt

from scipy.io import wavfile

 

# Read the wav file (mono)

#samplingFrequency, signalData = wavfile.read('/Users/Sebastian/Desktop/bird_mp3/wav/2018-06-09 23.50.40.wav')


rate, data = wavfile.read('/Volumes/500 GB/alex_25_05_18/files//Bubo_bubo/Bubo_bubo-410857_5.wav')
fig,ax = plt.subplots(1)
fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
ax.axis('on')
ax.set_xlabel("Frequency")
pxx, freqs, bins, im = ax.specgram(x=data[:,0], Fs=rate, noverlap=384, NFFT=512)
#pxx, freqs, bins, im = ax.specgram(x=data, Fs=rate, noverlap=384, NFFT=512)
ax.axis('on')

fig.savefig('./assets/spectrogram_uhu.png', dpi=300, frameon='true')

"""
print(signalData.shape)
	
plot.subplot(111)

plot.specgram(signalData[:,1],Fs=samplingFrequency)

plot.xlabel('Time')

plot.ylabel('Frequency')
"""
 

#plot.show()
