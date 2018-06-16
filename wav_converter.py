from pydub import AudioSegment
import os 


def converter():
	for x,subdirList, fileList in os.walk("./mp3/"):
		print(x)
		print(subdirList)
		print(fileList)
		for filename in fileList:
			if filename.endswith(".mp3"):
				path = str(x + filename)
				print(filename) 
				sound = AudioSegment.from_mp3(path)
				name, ext = os.path.splitext(filename)
				new_name = name
				#sound.export(str("./wav/"+str(name)), format="wav")
				sound.export(str("./wav/"+name+".wav"), format="wav")
				print("Finished: "+str(name))


print("starting")
converter()
print("program executed")