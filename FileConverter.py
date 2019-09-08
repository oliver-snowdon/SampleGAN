from AudioFileManager import *
import re
import os
import glob
import ntpath

if __name__ == "__main__":
	newSampleRate = 8000
	if not os.path.exists('Data'):
		os.mkdir('Data')
	for filename in glob.glob("Playlist/*.wav"):
		print(filename)
		if not os.path.exists("Data/{}.wav".format(ntpath.basename(filename))):
			data = ReadWavAsMonoResampled(filename, newSampleRate)
			WriteMonoWav("Data/{}.wav".format(ntpath.basename(filename)), data, newSampleRate)

	for filename in glob.glob("Playlist/*.mp3"):
		print(filename)
		if not os.path.exists("Data/{}.wav".format(ntpath.basename(filename))):
			data = ReadWavAsMonoResampled(filename, newSampleRate)
			WriteMonoWav("Data/{}.wav".format(ntpath.basename(filename)), data, newSampleRate)
