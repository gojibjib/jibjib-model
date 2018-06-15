from pydub import AudioSegment
import os 
import numpy as np
import librosa
from scipy.io.wavfile import read



def pitcher():
	for x,subdirList, fileList in os.walk("./wav/"):
		for filename in fileList:
			if filename.endswith(".wav"):
				path = str(x + filename)
				print(filename) 
				#sound = AudioSegment.from_wav(path)
				sound,_ = read(path)
				stretched = librosa.core.resample(sound, 4000, 16000)
				print("printing sound...")
				print(sound)
				name, ext = os.path.splitext(filename)
				streched = speedx(sound,factor)
				new_name = name
				#sound.export(str("./wav/"+str(name)), format="wav")
				stretched.export(str("./wav/"+name+"_stretched_"+str(factor)+".wav"), format="wav")
				print("Finished: "+str(name))



print("starting")
pitcher()
print("program executed")