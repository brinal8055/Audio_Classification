import pandas as pd
import numpy as np
import os
import struct
import librosa
from scipy.io import wavfile as wav
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint 
from datetime import datetime 
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics 


metadata = pd.read_csv('./Udacity-ML-Capstone/UrbanSound Dataset sample/metadata/UrbanSound8K.csv')
# metadata.head()

class WavFileHelper():
    
    def read_file_properties(self, filename):

        wave_file = open(filename,"rb")
        
        riff = wave_file.read(12)
        fmt = wave_file.read(36)
        
        num_channels_string = fmt[10:12]
        num_channels = struct.unpack('<H', num_channels_string)[0]

        sample_rate_string = fmt[12:16]
        sample_rate = struct.unpack("<I",sample_rate_string)[0]
        
        bit_depth_string = fmt[22:24]
        bit_depth = struct.unpack("<H",bit_depth_string)[0]

        return (num_channels, sample_rate, bit_depth)

wavfilehelper = WavFileHelper()

audiodata = []
for index, row in metadata.iterrows():
    
    file_name = os.path.join(os.path.abspath('./UrbanSound8K/audio/'),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
    data = wavfilehelper.read_file_properties(file_name)
    audiodata.append(data)

# Convert into a Panda dataframe
audiodf = pd.DataFrame(audiodata, columns=['num_channels','sample_rate','bit_depth'])

print('Num of channels: ', audiodf.num_channels.value_counts(normalize=True),'\n')
print('Sample rates: ',audiodf.sample_rate.value_counts(normalize=True),'\n')
print('Bit depth: ',audiodf.bit_depth.value_counts(normalize=True),'\n')

filename = './Udacity-ML-Capstone/UrbanSound Dataset sample/audio/100852-0-0-0.wav' 

librosa_audio, librosa_sample_rate = librosa.load(filename) 
scipy_sample_rate, scipy_audio = wav.read(filename) 

print('Original sample rate:', scipy_sample_rate) 
print('Librosa sample rate:', librosa_sample_rate,'\n') 

mfccs = librosa.feature.mfcc(y=librosa_audio, sr=librosa_sample_rate, n_mfcc=40)
# print('Shape of MFCCS ', mfccs.shape)

num_rows = 40
num_columns = 174
num_channels = 1

# 1. MLP
# 2. CNN
print('1. MLP')
print('2. CNN')
xif=input('Enter the implementation model number (1 or 2): ')

# MLP model
if xif=='1':
    def extract_features(file_name):
        try:
            audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
            mfccsscaled = np.mean(mfccs.T,axis=0)
            
        except Exception as e:
            print("Error encountered while parsing file: ", file_name)
            return None 
        
        return mfccsscaled

    # Set the path to the full UrbanSound dataset 
    fulldatasetpath = './UrbanSound8K/audio/'

    metadata = pd.read_csv('./Udacity-ML-Capstone/UrbanSound Dataset sample/metadata/UrbanSound8K.csv')

    features = []

    # Iterate through each sound file and extract the features 
    for index, row in metadata.iterrows():
        
        file_name = os.path.join(os.path.abspath(fulldatasetpath),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
        
        class_label = row["class_name"]
        data = extract_features(file_name)
        
        features.append([data, class_label])

    # Convert into a Panda dataframe 
    featuresdf = pd.DataFrame(features, columns=['feature','class_label'])

    # Convert features and corresponding classification labels into numpy arrays
    X = np.array(featuresdf.feature.tolist())
    y = np.array(featuresdf.class_label.tolist())

    # Encode the classification labels
    le = LabelEncoder()
    yy = to_categorical(le.fit_transform(y)) 

    x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state = 42)


    num_labels = yy.shape[1]
    filter_size = 2

    # Construct model 
    model = Sequential()

    model.add(Dense(256, input_shape=(40,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_labels))
    model.add(Activation('softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    # Display model architecture summary 
    model.summary()

    # Calculate pre-training accuracy 
    score = model.evaluate(x_test, y_test, verbose=0)
    accuracy = 100*score[1]

    print("Pre-training accuracy: %.4f%%" % accuracy)


    num_epochs = 100
    num_batch_size = 32

    checkpointer = ModelCheckpoint(filepath='./Udacity-ML-Capstone/Notebooks/saved_models/weights.best.basic_mlp.hdf5', 
                                verbose=1, save_best_only=True)
    start = datetime.now()

    model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), callbacks=[checkpointer], verbose=1)


    duration = datetime.now() - start
    print("Training completed in time: ", duration)

    # Evaluating the model on the training and testing set
    score = model.evaluate(x_train, y_train, verbose=0)
    print("Training Accuracy: ", score[1])

    score = model.evaluate(x_test, y_test, verbose=0)
    print("Testing Accuracy: ", score[1], '\n')

    def extract_feature(file_name):
    
        try:
            audio_data, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
            mfccsscaled = np.mean(mfccs.T,axis=0)
            
        except Exception as e:
            print("Error encountered while parsing file: ", file_name)
            return None, None

        return np.array([mfccsscaled])

    Classes = ['air_conditioner','car_horn','children_playing','dog_bark','drilling','engine_idling','gun_shot','jackhammer','siren','street_music']

    def print_prediction(file_name):
        prediction_feature = extract_feature('./test/'+file_name) 

        predicted_vector = model.predict(prediction_feature)
        predicted_class = np.argmax(predicted_vector)
        # predicted_class = le.inverse_transform(predicted_vector) 
        print("The predicted class is:", Classes[predicted_class], '\n')

# CNN model
elif xif=='2':

    max_pad_len = 174

    def extract_features(file_name):
    
        # try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
            
        # except Exception as e:
        #     print("Error encountered while parsing file: ", file_name)
        #     return None 
        
        return mfccs

    # Set the path to the full UrbanSound dataset 
    fulldatasetpath = './UrbanSound8K/audio/'

    metadata = pd.read_csv('./Udacity-ML-Capstone/UrbanSound Dataset sample/metadata/UrbanSound8K.csv')
    metadata = metadata.rename(columns={'class': 'class_name'})

    features = []

    # Iterate through each sound file and extract the features 
    for index, row in metadata.iterrows():
        
        file_name = os.path.join(os.path.abspath(fulldatasetpath),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
        
        class_label = row["class_name"]
        data = extract_features(file_name)
        
        features.append([data, class_label])

    # Convert into a Panda dataframe 
    featuresdf = pd.DataFrame(features, columns=['feature','class_label'])

    # Convert features and corresponding classification labels into numpy arrays
    X = np.array(featuresdf.feature.tolist())
    y = np.array(featuresdf.class_label.tolist())

    # Encode the classification labels
    le = LabelEncoder()
    yy = to_categorical(le.fit_transform(y)) 

    # split the dataset 
    x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state = 42)

    print(x_train.shape)
    x_train = x_train.reshape(x_train.shape[0], num_rows, num_columns, num_channels)
    x_test = x_test.reshape(x_test.shape[0], num_rows, num_columns, num_channels)

    num_labels = yy.shape[1]
    filter_size = 2

    # Construct model 
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=2, input_shape=(num_rows, num_columns, num_channels), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=128, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(GlobalAveragePooling2D())

    model.add(Dense(num_labels, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    # Display model architecture summary 
    model.summary()

    # Calculate pre-training accuracy 
    score = model.evaluate(x_test, y_test, verbose=1)
    accuracy = 100*score[1]

    print("Pre-training accuracy: %.4f%%" % accuracy)

    num_epochs = 72
    num_batch_size = 256

    checkpointer = ModelCheckpoint(filepath='./Udacity-ML-Capstone/Notebooks/saved_models/weights.best.basic_cnn.hdf5',verbose=1, save_best_only=True)
    start = datetime.now()

    model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), callbacks=[checkpointer], verbose=1)


    duration = datetime.now() - start
    print("Training completed in time: ", duration)

    # Evaluating the model on the training and testing set
    score = model.evaluate(x_train, y_train, verbose=0)
    print("Training Accuracy: ", score[1])

    score = model.evaluate(x_test, y_test, verbose=0)
    print("Testing Accuracy: ", score[1])

    Classes = ['air_conditioner','car_horn','children_playing','dog_bark','drilling','engine_idling','gun_shot','jackhammer','siren','street_music']

    def print_prediction(file_name):
        prediction_feature = extract_features('./test/'+file_name) 
        # print(prediction_feature)
        prediction_feature = prediction_feature.reshape(1, num_rows, num_columns, num_channels)

        predicted_vector = model.predict(prediction_feature)
        predicted_class = np.argmax(predicted_vector)
        # predicted_class = le.inverse_transform(predicted_vector) 
        print("The predicted class is:", Classes[predicted_class], '\n')

else:
    print('Enter 1 or 2  !!!')
    exit()

output = open("parameters.txt","w")
output.write(xif)
output.write(str(num_rows)+'\n')
output.write(str(num_columns)+'\n')
output.write(str(num_channels)+'\n')

model.save('./models/classification_model')

# # Class: Air Conditioner

# filename = './Udacity-ML-Capstone/UrbanSound Dataset sample/audio/100852-0-0-0.wav' 
# print_prediction(filename)

# # Class: Drilling

# filename = './Udacity-ML-Capstone/UrbanSound Dataset sample/audio/103199-4-0-0.wav'
# print_prediction(filename)

# # Class: Street music 

# filename = './Udacity-ML-Capstone/UrbanSound Dataset sample/audio/101848-9-0-0.wav'
# print_prediction(filename)

# # Class: Car Horn 

# filename = './Udacity-ML-Capstone/UrbanSound Dataset sample/audio/100648-1-0-0.wav'
# print_prediction(filename)

# # Class: Given

# filename = './Udacity-ML-Capstone/Evaluation audio/siren_1.wav'
# print_prediction(filename)