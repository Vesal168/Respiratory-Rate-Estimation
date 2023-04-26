import numpy as np
import scipy
import scipy.io.wavfile as wavfile

# Set constants
WINDOW_SIZE = 512
SAMPLE_RATE = 44100
LOW_PASS_FILTER = 100.0

# Load audio data from file
sampling_rate, audio_data = wavfile.read('Data/file.wav')

# Convert audio data to mono
if audio_data.ndim > 1:
    audio_data_mono = np.mean(audio_data, axis=1)
else:
    audio_data_mono = audio_data

# Apply a high-pass filter to remove low-frequency noise
audio_data_mono = np.asarray(audio_data_mono, dtype=np.float32)
audio_data_mono -= np.mean(audio_data_mono)
audio_data_mono /= np.max(np.abs(audio_data_mono))
b = np.array([1.0, -0.95])
audio_data_mono = np.convolve(audio_data_mono, b, mode='same')

# Apply a low-pass filter to remove high-frequency noise
nyquist_rate = sampling_rate / 2.0
cutoff_freq = LOW_PASS_FILTER / nyquist_rate
b, a = scipy.signal.butter(4, cutoff_freq, btype='low', analog=False)
audio_data_mono = scipy.signal.filtfilt(b, a, audio_data_mono)

# Split audio data into overlapping windows
num_windows = (len(audio_data_mono) - WINDOW_SIZE) // (WINDOW_SIZE // 2) + 1
audio_data_windows = np.zeros((num_windows, WINDOW_SIZE))
for i in range(num_windows):
    start_index = i * (WINDOW_SIZE // 2)
    end_index = start_index + WINDOW_SIZE
    audio_data_windows[i, :] = audio_data_mono[start_index:end_index]
