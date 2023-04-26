# Import necessary libraries:
import numpy as np
import scipy.signal as signal
from scipy.io import wavfile
from scipy.signal import decimate
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt


# 1. Load the audio signal using scipy.io.wavfile.read()

fs, audio = wavfile.read('Data/2023-04-20_19-14-01.wav')
print(f"Audio Original: {audio} and Sampling Rate: {fs}")

# 2. Apply a band-pass filter [500-5000] Hz to remove noise and baseline drift using scipy.signal.butter() and scipy.signal.filtfilt():

lowcut = 500  # Lower cutoff frequency (Hz)
highcut = 5000  # Upper cutoff frequency (Hz)
# fs = 44100  # Sampling frequency (Hz)
order = 4  # Filter order
nyquist = 0.5 * fs
low = lowcut / nyquist
high = highcut / nyquist

b, a = signal.butter(order, [low, high], btype='band')
audio_band_pass = signal.filtfilt(b, a, audio)
print(f"Audio after Applies a Band-Pass Filtered: {audio_band_pass}")

# 3. Find the Envelope of the signal using Hilbert Transform

amplitude_envelope  = np.abs(signal.hilbert(audio_band_pass))
print(f"Envelope of the signal: {amplitude_envelope}")

fs_new = 100
num_samples = int(np.floor(len(amplitude_envelope) / fs * fs_new))
envelope_digitized = np.zeros(num_samples)
for i in range(num_samples):
    start = int(i * fs / fs_new)
    end = int(start + fs / fs_new)
    envelope_digitized[i] = np.mean(amplitude_envelope[start:end])

# 4. Cubic Spline Interpolation @ fixed 100Hz

t = np.linspace(0, len(envelope_digitized) / 100, len(envelope_digitized))  # time points of the original data
# Find the envelope of the signal using Hilbert transform as shown in the previous step
y_hilbert = np.abs(signal.hilbert(envelope_digitized))
t_new1 = np.arange(0, len(y_hilbert) / 100, 1/100)  # time points of the resampled data
# Resample the signal to a fixed 100Hz rate using cubic spline interpolation
cs = CubicSpline(t, y_hilbert)
envelope_resampled  = cs(t_new1)
print(f"Cubic Spline Interpolation @ fixed 100Hz of Signal Resampleed{envelope_resampled }")

# 5. Apply a band-pass filter [0.19-4.6] Hz

nyquist = 0.5 * 100  # Nyquist frequency is half the sampling rate
lowcut = 0.19 / nyquist
highcut = 4.6 / nyquist
b, a = signal.butter(4, [lowcut, highcut], btype='band')
envelope_filtered = signal.filtfilt(b, a, envelope_resampled)
print(f"Band-pass filter [0.19-4.6] Hz: {envelope_filtered}")

# 6. crop the signal from index 100 to 1000
cropped_signal = envelope_filtered[100:1000]
print(f"crop the signal from index 100 to 1000: {cropped_signal}")

# 7. Down-sample to 10Hz
fs = 100  # original sampling frequency
fs_new = 10  # new sampling frequency
audio_ds = decimate(cropped_signal, q=int(fs / fs_new), zero_phase=True)
print(f"Down-sample to 10Hz: {audio_ds}")

# 8. Calculate the PSD using the welch function from scipy.signal
freqs, psd = signal.welch(audio_ds, fs=10, nperseg=1024)
print(f"Power Spectral Density: {psd}")
print(f"Frequency of Power Spectral Density: {freqs}")

# 9. Detect peaks in PSD
peaks1, _ = signal.find_peaks(psd, height=10)  # height is the minimum height of a peak
peak_frequencies = psd[peaks1]  # Get the corresponding frequencies
print(f"Peak of PSD: {peaks1}")
print(f"Peak of Frequency of Power Spectral Density: {peak_frequencies}")


# 10. Calculate the time intervals between the peaks to estimate the respiratory rate
# intervals = np.diff(peaks1) / fs
# rr = 60 / np.mean(intervals)

intervals = np.diff(peaks1)
rr = 60 / intervals
print(f"Time intervals: {intervals}")
print(f"Estimate the Respiratory Rate: {rr}")

# 11. Calculate median and IQR of respiratory rates
median_rate = np.median(rr)
iqr = np.percentile(rr, 75) - np.percentile(rr, 25)
print(f"Median of RR: {median_rate}")
print(f"Interquartilie Range of : {iqr}")
frequencies, times, spectrogram_data  = signal.spectrogram(audio, fs=fs)

# Visualization
# 1. Plot the Original Signal
fig, ax = plt.subplots()
ax.plot(audio, label='Audio Signal')
ax.set_xlabel('Time (samples)')
ax.set_ylabel('Amplitude')
ax.set_title('1. Plot the Original Signal')
ax.legend()

# Plot the spectrogram
# 1.1 Plot the magnitude
magnitude = 20 * np.log10(np.abs(spectrogram_data))
fig, ax = plt.subplots()
plt.pcolormesh(times, frequencies, magnitude)
ax.set_xlabel('Frequency [Hz]')
ax.set_xlabel('Time [sec]')
ax.set_title('1.1 Plot the spectrogram')
ax.legend()

# 2. Plot the Bandpass Signal [500-5000] Hz
fig, ax = plt.subplots()
ax.plot(audio, label='Audio Signal')
ax.plot(audio_band_pass, label='Signal Filtered')
ax.set_xlabel('Time (samples)')
ax.set_ylabel('Amplitude')
ax.set_title('2. Bandpass Filter Signal [500-5000] Hz')
ax.legend()

# 3. Plot Envelope of the signal
fig, ax = plt.subplots()
ax.plot(amplitude_envelope)
ax.set_xlabel('Time (samples)')
ax.set_ylabel('Amplitude')
ax.set_title('3. Envelope of the signal')
ax.legend()

# Plot the original and digitized signals
t = np.arange(len(amplitude_envelope)) / fs
t_new = np.arange(len(envelope_digitized)) / fs_new
fig, ax = plt.subplots(2, 1, figsize=(10, 6))
ax[0].plot(t, amplitude_envelope)
ax[0].set_xlabel('Time (s)')
ax[0].set_ylabel('Amplitude')
ax[0].set_title('Original Signal')
ax[1].plot(t_new, envelope_digitized)
ax[1].set_xlabel('Time (s)')
ax[1].set_ylabel('Amplitude')
ax[1].set_title('4. Digitized Signal @ 100Hz')
plt.tight_layout()

# 5. Cubic Spline Interpolation @ fixed 100Hz
fig, ax = plt.subplots()
plt.plot(t_new1, envelope_resampled)
ax.set_ylabel('Frequency [Hz]')
ax.set_xlabel('Time [sec]')
ax.set_title('5. Cubic Spline Interpolation @ fixed 100Hz')
ax.legend()

# 6. Plot a band-pass filter [0.19-4.6] Hz
fig, ax = plt.subplots()
ax.plot(envelope_filtered, label='Signal Filtered')
ax.set_xlabel('Time (samples)')
ax.set_ylabel('Amplitude')
ax.set_title('6. Bandpass Filter Signal [0.19-4.6] Hz')
ax.legend()

# 7. crop the signal from index 100 to 1000
fig, ax = plt.subplots()
ax.plot(cropped_signal)
ax.set_xlabel('Time (samples)')
ax.set_ylabel('Amplitude')
ax.set_title('7. crop the signal from index 100 to 1000')
ax.legend()

# 8. Down-sample to 10Hz audio_ds
fig, ax = plt.subplots()
ax.plot(audio_ds)
ax.set_xlabel('Time (samples)')
ax.set_ylabel('Amplitude')
ax.set_title('8. Down-sample to 10Hz')
ax.legend()

# 9. Plot the Power Spectral Density
fig, ax = plt.subplots()
ax.plot(freqs, psd)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Power Spectral Density')
ax.set_title('9. Plot the Power Spectral Density')
ax.legend()

# 10. Detect peaks in PSD
fig, ax = plt.subplots()
ax.plot(psd, "-", label='Power Spectral Density')
ax.plot(peaks1, peak_frequencies, "x", label='Peak Detection')
ax.set_xlabel('Time (samples)')
ax.set_ylabel('Amplitude')
ax.set_title('10. Detect peaks in Power Spectral Density')
ax.legend()

# 11. Intervals between the peaks to estimate the respiratory rate
fig, ax = plt.subplots()
ax.plot(rr, label='Respiratory Rate')
ax.set_xlabel('Time (samples)')
ax.set_ylabel('Respiration Rate (breaths per minute)')
ax.set_title('11. Intervals Between the Peaks to Estimate the RR')
ax.legend()
plt.show()