import numpy as np
import scipy.io.wavfile
import math
import librosa.core

def ReadWavAsStereo(filename):
	samples, rate = librosa.core.load(filename, None)
	if samples.ndim == 1:
		return rate, samples, samples
	else:
		transpose = np.transpose(samples)
		return rate, transpose[0], transpose[1]

def ReadWavAsMono(filename):
	samples, rate = librosa.core.load(filename, None)
	if samples.ndim == 1:
		return rate, samples
	else:
		transpose = np.transpose(samples)
		return rate, (transpose[0] + transpose[1]) / 2

def ReadWavAsMonoResampled(filename, targetRate):
	inputRate, samples = ReadWavAsMono(filename)
	if inputRate == targetRate:
		return samples
	else:
		return librosa.core.resample(samples, inputRate, targetRate)

def WriteMonoWav(filename, samples, sampleRate):
	samplesInt = np.zeros_like(samples, dtype='int16')
	samplesInt[:] = samples[:]*32767
	scipy.io.wavfile.write(filename, sampleRate, samplesInt)
