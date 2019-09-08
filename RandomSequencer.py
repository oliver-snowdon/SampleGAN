from AudioFileManager import *
import numpy as np

sequenceLength = 4775*2
nSnippets = 256
minEpoch = 61000
nEpochs = 800
epochStep = 10
snippetsPerEpoch = 5
nOverlays = 1
nRepeats = 4

def GetSequence():
	output = np.zeros(sequenceLength*nSnippets*nRepeats)
	for i in range(nSnippets):
		epoch = minEpoch + np.random.randint(0, nEpochs)*epochStep
		k = np.random.randint(0, snippetsPerEpoch)
		for j in range(nRepeats):
			rate, audio = ReadWavAsMono("GANOutputs/normalized{}.{}.wav".format(epoch, k))
			output[i*nRepeats*sequenceLength+j*sequenceLength:i*nRepeats*sequenceLength+j*sequenceLength+sequenceLength] = audio[0:sequenceLength]
	return output

output = np.zeros(sequenceLength*nSnippets*nRepeats)
for i in range(nOverlays):
	offset = sequenceLength * i
	output[offset:-1] = output[offset:-1] + GetSequence()[offset:-1]/nOverlays

WriteMonoWav("randomSequence.wav", output, 8000)
