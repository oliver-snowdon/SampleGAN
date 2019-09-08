# SampleGAN
Generative Adversarial Network for making Music Samples

# Instructions
1. Run `python FileConverter.py` to convert .wav and .mp3 files in the Playlist folder into .wav files of the correct sample rate in the Data folder.
2. Run `python SampleGAN.py` to start training the GAN and producing output.
3. When there is sufficient output, run `python RandomSequencer.py` to create a random arrangement of GAN outputs. Unless you modify the code, the criterion for "when there is sufficient output" will be when the GAN has been trained for at least 69000 epochs.
4. Listen to `randomSequence.wav` and be on the lookout for bits you like, or just use it as background noise.
