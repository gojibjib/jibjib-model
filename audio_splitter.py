#audio file splitter
#audio file augmentation

from pydub import AudioSegment
import os 

def splitter():
	for x,subdirList, fileList in os.walk("./wav/"):
		print (x)
		for filename in fileList:
			if filename.endswith(".wav"):
				path = str(x+filename)
				print(filename)
				sound = AudioSegment.from_wav(path)
				#over 30 seocnds
				if(len(sound)>15000):
					print(filename + " is: "+str(len(sound)))
					slices = sound[::9000]
					counter=1
					for element in slices:
						name, ext = os.path.splitext(filename)
						new_name = str(name+"_"+str(counter))
						element.export(str("./splitted/"+new_name+".wav"), format="wav")
						counter+=1




print("starting")
splitter()
print("program executed")