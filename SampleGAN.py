from keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization, Activation, Conv2DTranspose, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv1D, ZeroPadding1D
from keras.layers.merge import _Merge
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
import keras.backend as K
from functools import partial
import numpy as np

from AudioFileManager import *

def Conv1DTranspose(input_tensor, filters, kernel_size, strides=2, padding='same'):
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1), padding=padding)(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    return x

class RandomWeightedAverage(_Merge):
	def __init__(self, batchSize):
		self.batchSize = batchSize
		super(RandomWeightedAverage, self).__init__()

	"""Provides a (random) weighted average between real and generated samples"""
	def _merge_function(self, inputs):
		alpha = 1*K.random_uniform((self.batchSize, 1, 1))
		return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class GAN():
	def wasserstein_loss(self, y_true, y_pred):
		return K.mean(y_true * y_pred)

	def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
		"""
		Computes gradient penalty based on prediction and weighted real / fake samples
		"""
		gradients = K.gradients(y_pred, averaged_samples)[0]
		# compute the euclidean norm by squaring ...
		gradients_sqr = K.square(gradients)
		#   ... summing over the rows ...
		gradients_sqr_sum = K.sum(gradients_sqr,
					  axis=np.arange(1, len(gradients_sqr.shape)))
		#   ... and sqrt (add a small number first to prevent "nan" result)
		gradient_l2_norm = K.sqrt(K.clip(gradients_sqr_sum, 1e-12, 1e12))
		# compute lambda * (1 - ||grad||)^2 still for each single sample
		gradient_penalty = K.square(1 - gradient_l2_norm)
		# return the mean as loss over all the batch samples
		return K.mean(gradient_penalty)

	def __init__(self, length, fftLength, batchSize):

		self.batchSize = batchSize
		self.n_critic = 5

		self.outputShape = (length, 1)
		self.nRandom = 1000

		optimizerD = RMSprop(lr=0.00005)
		optimizerG = RMSprop(lr=0.00005)

		#Build the generator and critic
		self.critic = self.BuildCritic()
		self.generator = self.BuildGenerator()

		#Freeze generator's layers while training critic
		self.generator.trainable = False

		# Waveform input (real sample)
		real_wav = Input(shape=self.outputShape)

		# Noise input
		z_disc = Input(shape=(self.nRandom,))
		# Generate waveform based of noise (fake sample)
		fake_wav = self.generator(z_disc)

		# Discriminator determines validity of the real and fake waveforms
		fake = self.critic(fake_wav)
		valid = self.critic(real_wav)

		# Construct weighted average between real and fake waveforms
		interpolated_wav = RandomWeightedAverage(batchSize)([real_wav, fake_wav])
		# Determine validity of weighted sample
		validity_interpolated = self.critic(interpolated_wav)

		# Use Python partial to provide loss function with additional
		# 'averaged_samples' argument
		partial_gp_loss = partial(self.gradient_penalty_loss,
					  averaged_samples=interpolated_wav)
		partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

		self.critic_model = Model(inputs=[real_wav, z_disc],
					  outputs=[valid, fake, validity_interpolated])
		self.critic_model.compile(loss=[self.wasserstein_loss,
						self.wasserstein_loss,
						partial_gp_loss],
					  optimizer=optimizerD,
					  loss_weights=[1, 1, 10])

		# For the generator we freeze the critic's layers
		self.critic.trainable = False
		self.generator.trainable = True

		# Sampled noise for input to generator
		z_gen = Input(shape=(self.nRandom,))
		# Generate waveforms based of noise
		wav = self.generator(z_gen)
		# Discriminator determines validity
		valid = self.critic(wav)
		# Defines generator model
		self.generator_model = Model(z_gen, valid)
		self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimizerG)

	def BuildGenerator(self):

		model = Sequential()
		
		alpha = 0.01
		kernelSize = 8

		model.add(Dense(64*2*256, input_dim=self.nRandom))
		model.add(Activation("tanh"))
		model.add(Reshape((64*2, 256)))
		model.add(Lambda(lambda x: K.expand_dims(x, axis=2)))
		model.add(Conv2DTranspose(filters=1024, kernel_size=(kernelSize, 1), strides=(2, 1), padding='same'))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Activation("tanh"))
		model.add(Conv2DTranspose(filters=512, kernel_size=(kernelSize, 1), strides=(2, 1), padding='same'))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Activation("tanh"))
		model.add(Conv2DTranspose(filters=256, kernel_size=(kernelSize, 1), strides=(2, 1), padding='same'))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Activation("tanh"))
		model.add(Conv2DTranspose(filters=128, kernel_size=(kernelSize, 1), strides=(2, 1), padding='same'))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Activation("tanh"))
		model.add(Conv2DTranspose(filters=32, kernel_size=(kernelSize, 1), strides=(2, 1), padding='same'))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Activation("tanh"))
		model.add(Conv2DTranspose(filters=2, kernel_size=(kernelSize, 1), strides=(2, 1), padding='same'))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Activation("tanh"))
		model.add(Conv2DTranspose(filters=1, kernel_size=(kernelSize, 1), strides=(2, 1), padding='same'))
		model.add(Lambda(lambda x: K.squeeze(x, axis=2)))

		model.summary()

		noise = Input(shape=(self.nRandom,))
		output = model(noise)

		return Model(noise, output)

	def BuildCritic(self):
	
		print(self.outputShape)

		kernel_size = 8

		model = Sequential()
		
		model.add(Conv1D(32, kernel_size=kernel_size, strides=2, input_shape=self.outputShape, padding="same"))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dropout(0.25))
		model.add(Conv1D(64, kernel_size=kernel_size, strides=2, padding="same"))
		model.add(ZeroPadding1D(padding=((0,1))))
		model.add(BatchNormalization(momentum=0.8))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dropout(0.25))
		model.add(Conv1D(128, kernel_size=kernel_size, strides=2, padding="same"))
		model.add(BatchNormalization(momentum=0.8))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dropout(0.25))
		model.add(Conv1D(256, kernel_size=kernel_size, strides=2, padding="same"))
		model.add(BatchNormalization(momentum=0.8))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dropout(0.25))
		model.add(Flatten())
		model.add(Dense(1))
		
		model.summary()

		inputs = Input(shape=self.outputShape)
		validity = model(inputs)

		return Model(inputs, validity)

	def Train(self, epochs, wav, sample_interval=50):

		# Adversarial ground truths
		valid = -np.ones((self.batchSize, 1))
		fake =  np.ones((self.batchSize, 1))
		dummy = np.zeros((self.batchSize, 1)) # Dummy gt for gradient penalty

		for epoch in range(epochs):

			noise = np.random.normal(0, 1, (self.batchSize, self.nRandom))

			# ---------------------
			#  Train Discriminator
			# ---------------------

			# Select a random batch of waveforms
			selection = np.zeros((self.batchSize, self.outputShape[0], 1))
			gridSize = 4775
			for i in range(self.batchSize):
				r = np.random.randint((int(len(wav)-self.outputShape[0])/gridSize))
				selection[i,:,0] = wav[r*gridSize:r*gridSize+self.outputShape[0]]

			#WriteMonoWav("selection.wav".format(epoch), selection[i,:,0], 8000)

			noise = np.random.normal(0, 1, (self.batchSize, self.nRandom))

			for k in range(self.n_critic):

				# Generate a batch of new waveforms
				outputs = self.generator.predict(noise)

				# Train the critic
				d_loss = self.critic_model.train_on_batch([selection, noise], [valid, fake, dummy])
				
				if k == 0:
					criticWeights = []
					for i in range(len(self.critic.layers)):
						criticWeights.append(self.critic.layers[i].get_weights())
				

				# ---------------------
				#  Train Generator
				# ---------------------

				# Train the generator (to have the discriminator label samples as valid)
				g_loss = self.generator_model.train_on_batch(noise, valid)
			
			for i in range(len(self.critic.layers)):
				layerWeights = []
				self.critic.layers[i].set_weights(criticWeights[i])
				criticWeights.append(layerWeights)
			
			# Plot the progress
			print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))

			# If at save interval => save generated waveform samples
			if epoch % sample_interval == 0:
				self.Sample(epoch)

			if epoch % 500 == 0:
				self.critic_model.save("SavedModels/{}".format(epoch))

	def Sample(self, epoch):
		nSamples = 5

		noise = np.random.normal(0, 1, (nSamples, self.nRandom))
		outputs = self.generator.predict(noise)
		
		nFFTs = outputs.shape[1]
		fftLength = outputs.shape[2]
		
		for j in range(nSamples):
			wav = np.zeros(nFFTs*fftLength)
			
			for i in range(nFFTs):
				wav[i*fftLength:(i+1)*fftLength] = outputs[j,i,0]

			WriteMonoWav("GANOutputs/raw{}.{}.wav".format(epoch, j), wav, 8000)
			WriteMonoWav("GANOutputs/normalized{}.{}.wav".format(epoch, j), wav/np.max(np.abs(wav)), 8000)
		
if __name__ == '__main__':
	sequenceLength = 128*128
	fftLength = 1

	rate, wav = ReadWavAsMono("./Data/GANFodder1.wav")
	n = 0
	wav = wav[sequenceLength*0*fftLength+n:sequenceLength*16*fftLength+n]

	WriteMonoWav("recoveredWav.wav", wav, 8000)

	gan = GAN(sequenceLength, fftLength, batchSize=32)
	gan.Train(epochs=300000, wav=wav, sample_interval=10)
