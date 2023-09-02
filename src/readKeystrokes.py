import os
import sys
import json
import time

from copy import deepcopy

import numpy as np
import wavio
import matplotlib.pyplot as plt
from scipy import fft
import librosa

# common value for audio signals
SAMPLE_RATE = 44100

# plot spectrum of time-domain signal and its sampling frequency
def plotSpectrum(y,Fs):

    # length of the signal, number of samples in the signal
    n = len(y) 
    # array containing integers from 0 to n-1 to help calculate the frequency axis
    k = np.arange(n) 
    # duration of the signal in seconds
    T = n/Fs
    # two sides frequency range in Hz
    frq = k/T 
    # because of the two sides frequency range, we only take the first fequency range
    frq = frq[:2000]
    # fft computing and normalization
    # FFT is used to transform the signal from the time domain to the frequency domain
    Y = fft.fft(y)/n 
    # limit to the first 2000 points
    Y = Y[:2000]
    # return frequency axis and the magnitude of the FFT result (amplitude spectrum)
    return frq, abs(Y)

# find peaks in given signal (data)
# distance = 1 -> adjacent peaks are allowed
# threshold = 0.1 -> peaks must have greater value to be considered
def find_peaks(data, distance=1, threshold=0.1):
    peaks = []
    i = 0
    while i < len(data):
        # search for the local maximum within a specified range
        max_loc = np.argmax(data[i:i+distance]) + i
        # if the value at local maximum is greater than the threshold
        if data[max_loc] > threshold:
            # add local maximum to peaks
            peaks.append(max_loc)
            # move i to the next range to search for local max.
            i = max_loc + distance
        else:
            # skip a large portion of data to speed up the search
            i += 1000

    # return the list of peak indices
    return peaks



# File input (single WAV file path -> sound data encoded as array)

# read .wav file and return its data as a 1D NumPy array
def wav_read(filepath):

    # read the wave-formatted audio file using wavio library (returns a Wav object)
    data = wavio.read(filepath)
    # extract the data from the Wav object. The audio data is typically represented as a 2D NumPy array, where the first dimension represents the audio channels (e.g., stereo audio has two channels), and the second dimension represents the audio samples.
    data = data.data
    # check if the first element of data is a NumPy array (to see if the audio is multi-channel-stereo or mono-channel:single-channel)
    if type(data[0]) == np.ndarray:
        # if the audio is multi-channel, select only the first channel (convert multi to mono)
        data = data[:,0]
    # convert data type to np.float32 which is commonly used data type for audio data in NumPy
    data = data.astype(np.float32)
    # normalize the audio data by dividing it with the abs.max value in data to scale the audio in the range [-1,1]
    data = (data / np.max(np.abs(data)))
    # subtract the average value of the audio data from all samples to center the audio data around zero
    data -= np.mean(data)
    # return processed audio data as a 1D NumPy array
    return data

# detect and extract keystrokes from a sound data input
def detect_keystrokes(sound_data, sample_rate=SAMPLE_RATE, output=True, num_peaks = None, labels = None):
    
    # expected duration of a single keystroke in seconds
    keystroke_duration = 0.2 
    # calculate the number of samples required to represent the specified keystroke duration at the given sample rate (distance between peaks)
    len_sample = int(sample_rate * keystroke_duration)
    keystrokes = []

    # detect peaks in the sound data with given distance between keystrokes and a threshold
    peaks = find_peaks(sound_data, threshold=0.06, distance=len_sample)
    print(f"Found {len(peaks)} keystrokes in data")

    # if is not provided set labels as a list of None values with the same length as peaks list
    if not num_peaks:
        labels = [None for i in range(len(peaks))]

    # iterate through detected peaks
    for i, peak in enumerate(peaks):
        # for each peak adjust the peak position to get the beginning of the peak
        # most max_loc of the peaks are around the 1440 position so this number universally catches the beginning of the peak
        peak = peak - 1440
        # set the start and end positions for a fixed-duration segment of 0.2s around the peak
        # set start as peak
        # set end as peak + number of samples needed to represent a 0.2s duration at given sample rate
        start, finish = peak, peak + int(0.2 * sample_rate)

        # if the calculated end of the peak is beyond the length of the data set it to the length of data
        if finish > len(sound_data):
            finish = len(sound_data)

        print("Time in which the peak (keystroke) occured in seconds: ", peak/SAMPLE_RATE)

        # save the 0.2s segment of the keystroke as a keystoke
        keystroke = sound_data[start:finish]
        # append the audio segment of the keystroke as a list and the corresponding label of the detected keystroke
        keystrokes.append((keystroke.tolist(), labels[i]))
    
    return keystrokes


# Display detected keystrokes (WAV file -> all keystroke graphs)
def visualize_keystrokes(filepath, labels=None, default_label='unknown'):
    print("------- VISUALIZE KEYSTROKES --------")

    # read the given .wav file
    wav_data = wav_read(filepath)
    # detect keystrokes for the read .wav file
    keystrokes = detect_keystrokes(wav_data)
    #n = 1
    n = len(keystrokes)
    print('Drawing keystrokes...')
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # for each keystroke plot waveform of the 0.2s keystroke and its frequency spectrum
    for i in range(n):
        label = labels[i] if labels is not None else keystrokes[i][1]
        if labels is None:
            labels = label

        plt.figure(figsize=(12, 4))
        
        # Plot waveform
        plt.subplot(131)
        keystrokes_data = np.array(keystrokes[i][0])
        times = np.linspace(0, 0.2, len(keystrokes_data))
        plt.plot(times, keystrokes_data, color = 'darkorange')
        plt.xlabel('Vrijeme [s]', weight = 'bold')
        plt.ylabel('Amplituda', weight = 'bold')
        plt.title(f'Vrhovi od znaka ({label})', weight = 'bold')
        
        # Plot frequency spectrum
        plt.subplot(132)
        frq, y = plotSpectrum(np.array(keystrokes[i][0]), SAMPLE_RATE)
        plt.plot(frq, y, color = 'darkgreen')
        plt.xlabel('Frekvencija (Hz)', weight = 'bold')
        plt.ylabel('Veličina', weight = 'bold')
        plt.title(f'Frekvencijski spektar znaka ({label})', weight = 'bold')
        
        # plot log-frequency spectrogram of the keystroke
        # n_fft = 2048, hop_length = 512 for better frequency resolution
        plt.subplot(133)
        S = librosa.stft(np.array(keystrokes[i][0]), n_fft=2048, hop_length=512)
        x = np.abs(S) ** 2
        x_log = librosa.power_to_db(x)
        librosa.display.specshow(x_log, sr = SAMPLE_RATE, hop_length=512, x_axis="time", y_axis="log")
        plt.colorbar(format='%+2.0f')
        plt.xlabel('Vrijeme [s]', weight = 'bold')
        plt.ylabel('Frekvencija (Hz)', weight = 'bold')
        plt.title(f'Log-Frekvencijski spektrogram ({label})', weight = 'bold')

        plt.tight_layout()  # Ensure subplots don't overlap
        plt.savefig(f"..\\graphs\\waveform_and_freq_spec_{label}_{timestamp}_final2.png")
        plt.show()
        plt.close()
    plt.show()

# read character labels from a text file and return number of characters read and a list of characters
def getLabels(txtpath):
    # if txtpath is not empty or None
    if txtpath:
        # open and read the file and remove any newline characters from the data
        with open(txtpath, 'r') as file:
            data = file.read().strip('\n')
        # return number of characters and a list of characters
        return len(data), [char for char in data]
    else:
        return None, None



def main():
    # get the first argument from the command line
    filepath = str(sys.argv[1])
    # initialize text file path
    txtpath = None

    # if there is a second command line argument, save it as a txtpath
    if (len(sys.argv) > 2):
        txtpath = str(sys.argv[2])
    # generate an output file path for storing results, change it to the folder you want to save the results
    outfile = os.path.join("..\out", "keystrokes\\no_label", filepath.split("\\")[-1] + "_out")

    # read the .wav file
    wav_data = wav_read(filepath)

    # create a time axis of the audio signal
    x_axis = [(i/SAMPLE_RATE) for i in range(len(wav_data))]
    # plot the waveform of the whole audio signal
    plt.plot(x_axis, wav_data)
    plt.xlabel("Vrijeme [s]")
    plt.ylabel("Amplituda")
    # create a name for the plotted graph and save it to the full_signal_graph folder
    title = filepath.split("\\")[-1]
    title = "fullsignal_" + title + ".png"
    plt.savefig("..\\full_signal_graph\\"+title)
    # show the graph on screen
    plt.show()
    plt.close()

    # extract the characters(number of peaks = keystrokes) and labels from the .txt file
    num_peaks, labels = getLabels(txtpath)

    # detect keystrokes from the audio signal with given number of keystrokes and labels
    keystrokes = detect_keystrokes(wav_data, num_peaks = num_peaks, labels = labels)
    # check if the length of detected keystrokes matches the length of labels
    if labels is not None: 
        if(len(keystrokes) == len(labels)):
            print(f"Keystrokes are the same length as labels is...: {len(keystrokes) == len(labels)}")

    # write the detected keystrokes in JSON format to the output file
    with open(outfile, 'w') as f:
        f.write(json.dumps(keystrokes))

    # call visualize_keystrokes to see the visual representation of every keystroke and its frequency spectrum
    visualize_keystrokes(filepath, labels)

main()
