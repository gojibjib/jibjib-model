from pydub import AudioSegment
import os 
import numpy as np
import librosa
from scipy.io.wavfile import read



def pitcher():
	for x,subdirList, fileList in os.walk("/Users/Sebastian/Desktop/bird_mp3/wav/"):
		for filename in fileList:
			if filename.endswith(".wav"):
				path = str(x + filename)
				print(filename) 
				#sound = AudioSegment.from_wav(path)
				sound = AudioSegment.from_file(path, format="wav")

				name, ext = os.path.splitext(filename)
				
				octaves = [-1,-0.5,0.5,1]
				for element in octaves:
					new_sample_rate = int(sound.frame_rate * (2.0 ** octaves))

					# keep the same samples but tell the computer they ought to be played at the 
					# new, higher sample rate. This file sounds like a chipmunk but has a weird sample rate.
					hipitch_sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_sample_rate})

					# now we just convert it to a common sample rate (44.1k - standard audio CD) to 
					# make sure it works in regular audio players. Other than potentially losing audio quality (if
					# you set it too low - 44.1k is plenty) this should now noticeable change how the audio sounds.
					hipitch_sound = hipitch_sound.set_frame_rate(44100)
					#sound.export(str("./wav/"+str(name)), format="wav")
					hipitch_sound.export(str("./wav/"+name+"_pitched_"+str(octaves)+".wav"), format="wav")
				print("Finished: "+str(name))



print("starting")
pitcher()
print("program executed")