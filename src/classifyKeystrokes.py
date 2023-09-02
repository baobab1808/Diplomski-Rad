import sys
import os
import datetime
import torch
import json
import copy
import datetime
import operator
import time

import numpy as np
from librosa.feature import mfcc, melspectrogram
#from librosa.display import specshow
#from librosa import power_to_db
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
from collections import defaultdict
from pathlib import Path
#from torchviz import make_dot



class KeyDataLoader(Dataset):
    """Dataloader."""

    # defines a constructor which takes in a list of file paths as input and processes the data from these files to build a keystrokes list
    # called when an instance of the class is created
    def __init__(self, files):

        self.keystrokes = []
        # read each file one by one and extract data from them
        for file in files:
            #print(file)
            with open(file, "r") as f:
                # convert read files from json formatted string to Python data structure (list of tuples)
                data = json.loads(f.read().strip('\n'))
                # tuples with:
                # self.extract_features(np.array(a)) -> extract features from the NumPy array
                #data = [(self.extract_features(np.array(a), b),  self.convertletter(b)) for (a, b) in data]

                data = [(self.extract_features(np.array(a), b), self.convertletter(b)) if b is not None else self.extract_features_no_labels(np.array(a)) for (a, b) in data]

                # append newly created tupples to keystrokes list
                self.keystrokes.extend(data)


    # return the length of the keystrokes attribute
    def __len__(self):
        return len(self.keystrokes)

    # to customize the behavior of accessing elements from an instance of the class
    # return the item at the given index from the keystrokes attribute of the class instance
    def __getitem__(self, idx):
        return self.keystrokes[idx]

    # convert lowercase letter into a corresponding numeric representation using the following rules:
    # 1. if the input letter is a lowercase alphabet letter, the method converts it to a numeric value between 0 and 25 which represent the position of the letter in the English alphabet
    # 2. if the input letter is not a lowecase alphabet letter, the method assigns a special numeric value of 26
    def convertletter(self, l):
        # the difference between given letter and letter 'a' gives the index of the given letter in the English alphabet
        i = ord(l) - ord('a')
        if i < 0 or i > 25:
            i = 26
        # return a PyTorch LongTensor with a single value i and specifies the data type as long
        return torch.tensor(i).long()

    # extract a Mel-frequency ceostral coefficients (MFCC)-based feature vector from a given audio signal
    # input:
    # keystroke -> An audio signal, represented as a NumPy array, that will be used to compute the MFCC-based feature vector
    # sr -> The sampling rate of the audio signal (in Hz). The default value is 44100 Hz, which is a common value for audio signals
    # n_mfcc -> The number of MFCC coefficients to compute. The default value is 16
    # n_fft -> The number of data points used in each short-time Fourier transform (STFT) window. The default value is 220, which corresponds to a 10ms window for a 44100 Hz sampling rate
    # hop_len -> The number of data points to hop between consecutive STFT windows. The default value is 110, which corresponds to approximately 2.5ms hops for a 44100 Hz sampling rate
    
    def extract_features(self, keystroke, label, sr=44100, n_mfcc=16, n_fft=220, hop_len=110):
        '''Return an MFCC-based feature vector for a given keystroke.'''
        # uses the librosa.feature.mfcc function from the librosa library to compute the MFCC-based feature vector for the input audio signal (keystroke).
        # The mfcc function takes the following parameters:
        # y: The input audio signal as a NumPy array.
        # sr: The sampling rate of the audio signal.
        # n_mfcc: The number of MFCC coefficients to compute.
        # n_fft: The number of data points used in each STFT window.
        # hop_length: The number of data points to hop between consecutive STFT windows.
        # The mfcc function returns a 2D array (NumPy array) where each row represents the MFCC coefficients for a different frame of the audio signal.
        spec = mfcc(y=keystroke.astype(float),
                    sr=sr,
                    n_mfcc=n_mfcc,
                    n_fft=n_fft, # n_fft=220 for a 10ms window
                    hop_length=hop_len, # hop_length=110 for ~2.5ms
                    )
        
        """ To plot out the MFC coefficients uncomment this chunck of code and import for from librosa.display in size: 81x16"""
        """l = label
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        x1 = np.linspace(0, 0.2, 81)
        plt.figure(figsize=(10, 4))
        specshow(spec, x_axis='time', x_coords=x1, sr=sr, hop_length=hop_len, n_fft=n_fft)
        plt.colorbar(format='%+2.0f')
        plt.title(f'MFCC Koeficijenti za ({l})')
        plt.xlabel('Vrijeme (s)')
        plt.ylabel('MFCC Koeficijenti')
        plt.savefig(f"..\\graphs\\mel\\mfcc_{l}_{timestamp}_final.png")
        plt.show()
        plt.close()"""
        
        """ To create a mel-spectrogram and plot it with MFCCs, uncomment this chunck of code and imports for from librosa.display and librosa and subplot with MFCCs"""
        """mel_spectrogram = melspectrogram(
            y=keystroke.astype(float), 
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_len,
            n_mels=40  # Number of mel bands
            )

        # Convert to dB scale
        mel_spectrogram_db = power_to_db(mel_spectrogram)

        l = label
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        x1 = np.linspace(0, 0.2, 81)
        x2 = np.linspace(0, 0.2, 81)

        plt.figure(figsize=(12, 4))

        plt.subplot(121)
        specshow(spec, x_axis='time', x_coords=x1, sr=sr, hop_length=hop_len, n_fft=n_fft)
        plt.colorbar(format='%+2.0f')
        plt.title(f'MFCC Koeficijenti za ({l})')
        plt.xlabel('Vrijeme (s)')
        plt.ylabel('MFCC Koeficijenti')

        plt.subplot(122)
        specshow(mel_spectrogram_db, x_axis='time', y_axis='mel', sr=sr, x_coords=x2, hop_length=hop_len, n_fft=n_fft)
        plt.colorbar(format='%+2.0f dB')
        plt.xlabel('Vrijeme (s)')
        plt.title(f'Mel Spektrogram za ({l})')
        plt.tight_layout()
        plt.savefig(f"..\\graphs\\mel\\mfcc_mel_spec_{l}_{timestamp}_40_final.png")

        plt.show()
        plt.close()"""

        #The method flattens the 2D array using spec.flatten(), converting it into a 1D array (a single-dimensional list) that represents the MFCC-based feature vector.
        # Finally, the method converts the 1D NumPy array into a PyTorch tensor using torch.tensor(spec.flatten()) and returns this tensor as the output
        return torch.tensor(spec.flatten())
    
    # same function like extract_features, but no labeled data is sent
    def extract_features_no_labels(self, keystroke, sr=44100, n_mfcc=16, n_fft=220, hop_len=110):
        spec = mfcc(y=keystroke.astype(float),
                    sr=sr,
                    n_mfcc=n_mfcc,
                    n_fft=n_fft, # n_fft=220 for a 10ms window
                    hop_length=hop_len, # hop_length=110 for ~2.5ms
                    )
        return torch.tensor(spec.flatten())

# convert a numeric value into a lowercase alphabet letter
# if the input number is not within the range 0 to 25, return a space as a placeholder
def convertnumber(n):
    # convert the calculated ASCII code to a lowercase letter
    i = chr(n + ord('a'))
    # if not a lowercase letter, set i as space
    if n < 0 or n > 25:
        i = " "
    return i

class KeyNet(nn.Module):
    # initialize a neural network architecture with three fully connected layers
    def __init__(self):
        # call the constructor of the parent class, initializing the base class
        super(KeyNet, self).__init__()
        # first fully connected layer (1296 inputs and 256 outputs): output->tensor of size(batch_size,256)
        self.fc1 = nn.Linear(1296, 256)
        # second layer (256 inputs and 64 outputs)
        self.fc2 = nn.Linear(256, 64)
        # third layer (64 inputs and 27 outputs)
        self.fc3 = nn.Linear(64, 27)
        self.double()

    # forward pass of the neural network, describing ho the input data flows through the defined layers during inference or training
    # x -> input data in the neural network
    def forward(self, x):
        # data is first passed through the first fully connected layer and the output is passed through the ReLU function
        x = F.relu(self.fc1(x))
        # the result from the first layer passed through the second fully connected layer and the output is passed trhough the ReLU function
        x = F.relu(self.fc2(x))
        # output from the second layer is passed through the third layer and produces the final output of the neural network
        x = self.fc3(x)
        # return the prediction/output of the neural network
        return x


class ClassifyKeystrokes:
    def __init__(self, files, outfile, shuffle=True):

        # create and instance of the KeyDataLoader class from the given file
        self.dataset = KeyDataLoader(files)
        # split dataset to 80%/20% (training/validation)
        validation_split = 0.2

        # get the size of the dataset
        dataset_size = len(self.dataset)
        # create a list of indices from 0 to dataset_size - 1 (positions of data points in the dataset)
        indices = list(range(dataset_size))
        # calculate the number of data points to be included in the validation set
        split = int(np.floor(validation_split * dataset_size))

        if shuffle == True:
            # randomize the order of the data points
            np.random.shuffle(indices)
        # split the shuffled indices into training and validation set
        train_indices, val_indices = indices[split:], indices[:split]

        # Creating PyTorch data samplers and loaders:
        # create random subset sampler to shuffle the data during training
        # create data loader to load and iterate through datasets
        # -> associate it with the dataset but use the made samplers
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        self.trainloader = DataLoader(self.dataset, sampler=train_sampler)
        self.validloader = DataLoader(self.dataset, sampler=valid_sampler)

        # initialize a neural network model
        self.net = KeyNet()

        # if the pre-trained model exists loaded it, otherwise perform model training and save it
        """this model exists, so if you wish to train your own, just change the name here"""
        model2 = "..\\out\\raw_sentence\\models\\fc_nn_model_95_oba.txt"
        if os.path.exists(model2):
            self.net.load_state_dict(torch.load(model2))
        else:
            self.classify()
            self.savemodel()


        # create an instance of the KeyDataLoader class
        valid = KeyDataLoader(files)
        
        # iterate over the files
        for i, path in enumerate(files):
            #print(i, path)

            # open every file and read it, also remove any unwanted newline characters
            with open(path, 'r') as file:
                self.labels  = file.read().strip('\n')
            
            # create dataloader
            self.validloader = DataLoader(valid)

            # call the validate method
            self.validate()

    # method for training the neural network and optimizing it
    def classify(self):
        '''Classify keystrokes'''
        print("Training neural network...")
        # set up the loss function
        criterion = nn.CrossEntropyLoss()
        # set up the optimizer
        optimizer = optim.Adam(self.net.parameters(), lr=0.000005)
        best_acc = 0
        best_model = None
        loss = 0

        # loop over the dataset multiple times (250 epochs)
        for epoch in range(250):
            running_loss = 0.0
            # iterate over mini-batches of data from trainloader
            for i, data in enumerate(self.trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                # if data is a list of [inputs, labels] tensors get them both, else get only inputs

                if(len(data) == 2):
                    inputs, labels = data
                else:
                    inputs = data
                    labels = -1
                
                # zero the parameter gradients to prepare for backpropagation
                optimizer.zero_grad()

                # forward + backward + optimize
                # neural networks prediction
                outputs = self.net(inputs)
                # loss between predictions and true labels calculated using cross-entropy loss
                loss = criterion(outputs, labels)
                # compute gradients with backpropagation
                loss.backward()
                # update the neural network's weights based on computed gradients
                optimizer.step()

                """ To draw a diagram of the neural network, uncomment this line, and the import for torchviz"""
                """make_dot(outputs, params=dict(list(self.net.named_parameters()))).render("..\\out\\slika", format="png")"""

                
                # print statistics
                running_loss += loss.item()
                # print every 200 mini-batches
                if i % 200 == 199:    
                    print('[epoch: %d, batch size: %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 200))
                    running_loss = 0.0
            # after each epoch call validate to evaluate the accuracy of the updated neural network
            running_acc = self.validate()

            # update the best accuracy and best model
            if running_acc > best_acc:
                best_acc = running_acc
                best_model = copy.deepcopy(self.net)

        # update self.net with best model
        self.net = best_model

        print(f'Accuracy of the best model after all epochs is: {best_acc}%')

        """ To print neural network summary and parameters, uncomment this chunck of code"""
        print(best_model)

        modules = [module for module in best_model.modules()]
        params = [param.shape for param in best_model.parameters()]

        # Print Model Summary
        print(modules[0])
        total_params=0
        for i in range(1,len(modules)):
            j = 2*i
            param = (params[j-2][1]*params[j-2][0])+params[j-1][0]
            total_params += param
            print("Layer",i,"->\t",end="")
            print("Weights:", params[j-2][0],"x",params[j-2][1],
                "\tBias: ",params[j-1][0], "\tParameters: ", param)
        print("\nTotal Params: ", total_params)

        print('Finished Training')

    # method for evaluating the accuracy of the trained neural network on validation set
    def validate(self):
        # initialize counters for correctly classified instances and total number of instances
        correct = 0
        total = 0
        # disable gradient computation
        with torch.no_grad():
            # iterate over mini-batches of data
            for data in self.validloader:
                # extract the images (input data) and labels (true labels) from the data if data has both, else extract only images
                if(len(data) == 2):
                    images, labels = data
                else:
                    images = data
                    labels = -1
                # neural network makes a predictions based on the input data
                outputs = self.net(images)

                # find the predicted class (one with the highest predicted probability) for each prediction instance
                _, predicted = torch.max(outputs.data, 1)

                # update the total number of instances
                total += 1
                # update the correct counter by adding the number of correct instances where the predicted class matches the ground truth (labels)
                correct += (predicted == labels).sum().item()

        # calculate the accuracy of the model (in %)
        acc = 100 * correct / total
        print(f'Accuracy of the network for all files on the {len(self.validloader)} test keys: {acc}%')
        # return the calculated accuracy
        return acc
    
    # method for performing validation and prediction on a set of input files
    def validate_each_file(self, files, outfile):
        # iterate through files
        for i, path in enumerate(files):
            print(i, path)
            # for each file, open it and read its contents and store them in the labels, removing unwanted newline characters
            with open(path, 'r') as file:
                self.labels  = file.read().strip('\n')
            # set up a dataloader for the current file by creating a new instance of the KeyDataLoader with that file path
            self.validloader = DataLoader(KeyDataLoader([path]))
            
            # call validate_sentence method with current file path
            #acc, sentence, char_predict = self.validate_sentence(path)
            sentence = self.validate_sentence(path)

            # construct an output file path
            output_file = outfile + str(i) + "_raw.txt"
            # save the received predicted sentence in th output file
            with open(output_file, 'w') as f:
                f.write(sentence)
            
    # method for performing validation and prediction on a single input files
    def validate_sentence(self, path):
        # initialize a nested defaultdict with outer defaultdict with any type keys (e.g. characters) and the inner defaultdict has keys of any type with integer value, all initially set to 0
        char_predict = defaultdict(lambda: defaultdict(int))
        # initialize needed variables
        correct = 0
        total = 0
        sentence = ""
        correctly_predicted_chars = {}
        full_labels = []
        # disable gradient computation
        with torch.no_grad():
            # iterate over mini-batches of data
            for data in self.validloader:
                # extract the images (input data) and labels (true labels) from the data if data has both, else only extract images
                if(len(data) == 2):
                    images, labels = data
                else:
                    images = data
                    labels = -1
                # neural network makes a predictions based on the input data
                outputs = self.net(images)

                # append the outputs to full_labels list
                full_labels.append(outputs)

                # find the predicted class (one with the highest predicted probability) for each prediction instance
                _, predicted = torch.max(outputs.data, 1)

                if(labels != -1):
                # update char_predict dictionary to keep track of the frequency of each character's prediction
                    char_predict[convertnumber(labels.data)][convertnumber(predicted)] += 1
                # append predicted character to the sentence string
                sentence += convertnumber(predicted)
                # update the total number of instances
                total += 1
                # update the correct counter by adding the number of correct instances where the predicted class matches the ground truth (labels)
                correct += (predicted == labels).sum().item()

        # calculate the accuracy of the model (in %)
        acc = 100 * correct / total
        print(f'Accuracy of the network for one file on the {len(self.validloader)} test keys: {acc}%')
        # print out predicted sentence for each file
        print(f"Predicted sentence for {path}:")
        print(sentence)
        print("")

        # create a higher-dimensional array of full_labels
        full_labels = np.stack(full_labels, axis = 0)
        # call correct_with_prob method
        final = self.correct_with_prob(full_labels)
        # iterate through char_predict dictionary
        for char in char_predict:
            # contains a list of predicted characters and their frequency, sorted in ascending order of frequency
            sorted_char = sorted(char_predict[char].items(), key=operator.itemgetter(1))
            # print out most common prediction mistakes if the length of sorted_char is greater than 1.
            if(len(sorted_char) > 1):
                print(f"Most common mistake for char {char}: {sorted_char}")
            # if the length is 1, the character has been 100% correctly predicted, save the char and sorted_char to correctly_predicted_chars dictionary
            elif(len(sorted_char) == 1):
                correctly_predicted_chars[char] = sorted_char
            # print out 100% correcty predicted characters
        for char, sorted_char in correctly_predicted_chars.items():
            print(f"Character has been 100% correctly predicted {char}: {sorted_char}")
        print("")
        # return predicted sentence
        return sentence
        

    # method for saving the best trained model in a given path
    def savemodel(self):
        path = f"..\\out\\raw_sentence\\models\\fc_nn_model_{datetime.datetime.now().strftime('%dT%mT%yT%H%M%S')}.txt"
        print(f'Saving model in {path}')
        # save the parameters of th PyTorch neural network model to a file
        torch.save(self.net.state_dict(), path)

    # method for processing a list of labels and performs post-processing on them based on their probabilities
    def correct_with_prob(self, labels):
        # create a timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        # initialize needed variables
        sentence = []
        cur_word = []
        # loop through labels
        for i in range(len(labels)):
            # call the convertnumber method on the highest predicted probability (to determine which character it is)
            # and if it is a space character
            if (convertnumber(np.argmax(labels[i])) == " "):
                # assume it is the end of the word and append the current word to the sentence
                sentence.append(cur_word)
                # reset current word to an empty list
                cur_word = []
            # if the label is not a space character
            else:
                # select the top 5 character predictions with the highest probabilities in descending order (highest is the first)
                top_indices = np.flip(np.argsort(labels[i])[0])[:5]
                # convert the indices to characters with convertnumber method
                top_letters = [convertnumber(i) for i in top_indices]
                # append the characters to the current word
                cur_word.append(top_letters)

        # append the last current word to the sentence
        sentence.append(cur_word)
        # create an output file
        outfile = "..\\out\\raw_sentence\\out_"+timestamp+"_text.json"
        # save the corrected sentence, represented as a list of words with all 5 top characters there, as a JSON file
        # Note: this file is used for either manual refinement of the predicted sentence or to send to a language processing model
        with open(outfile, 'w') as f:
            json.dump(sentence, f)


def main():
    '''Run learn keystrokes'''

    """ if you want to load only one file and not a folder, uncomment line 513 and comment out lines: 515, 517, 523 and 525"""
    #files = sys.argv[1:]
    # initialize the files array
    files = []
    # state the source directory where all the files with data are stored, change it to the folder you want to run
    source_dir = Path('..\\out\\keystrokes\\no_label\\')
    # create an outfile
    outfile = os.path.join("..\\out", "raw_sentence", "sentence"+str(datetime.datetime.now().strftime('%dT%mT%yT%H%M%S')))
    print("--- Classify Keystrokes ---")

    # iterate through the source directory
    for file in source_dir.iterdir():
        # append each file to the files array
        files.append(file)

    # create an instance of the ClassifyKeystrokes class with given files and outfile
    classifier = ClassifyKeystrokes(files, outfile)
    # call method validate_each_file on the classifier object with given files and outfile
    classifier.validate_each_file(files, outfile)



main()
