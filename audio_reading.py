import pyaudio
import wave
from pandas.core.accessor import DirNamesMixin
from tensorflow import keras
import numpy as np
import librosa
import sys
import os

# Classification prediction
input = open("parameters.txt","r")
modelp = input.readline()[:-1]
num_rows = int(input.readline()[:-1])
num_columns = int(input.readline()[:-1])
num_channels = int(input.readline()[:-1])

a = 'MLP' if modelp == '1' else 'CNN'
print('Classification using',a,'\n\nClasses:')

Classes = ['air_conditioner','car_horn','children_playing','dog_bark','drilling','engine_idling','gun_shot','jackhammer','siren','street_music']
for i in Classes:
    print(i)
print('\n')
model = keras.models.load_model('./models/classification_model')

if modelp == '1':
    def extract_feature(file_name):
    
        try:
            audio_data, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
            mfccsscaled = np.mean(mfccs.T,axis=0)
            
        except Exception as e:
            print("Error encountered while parsing file: ", file_name)
            return None, None

        return np.array([mfccsscaled])

    def print_prediction(file_name):
        prediction_feature = extract_feature(sys.argv[1]+'/'+file_name) 

        predicted_vector = model.predict(prediction_feature)
        predicted_class = np.argmax(predicted_vector)
        match = (predicted_vector[0][predicted_class]/np.sum(predicted_vector))*100
        print("The predicted class is:", Classes[predicted_class], 'with', match, '%', ' accuracy.\n')
else:
    max_pad_len = 174

    def extract_features(file_name):
    
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width < 0:
            mfccs = mfccs[:,:max_pad_len]
        else:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
            
        return mfccs

    def print_prediction(file_name):
        prediction_feature = extract_features(sys.argv[1]+'/'+file_name)
        prediction_feature = prediction_feature.reshape(1, num_rows, num_columns, num_channels)

        predicted_vector = model.predict(prediction_feature)
        predicted_class = np.argmax(predicted_vector)
        match = (predicted_vector[0][predicted_class]/np.sum(predicted_vector))*100
        print("The predicted class is:", Classes[predicted_class], 'with', match, '%', ' accuracy.\n')


# Recording prediction
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

print("Recording started")

flag=0

while flag==0:
    frames = []

    for i in range(int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Recorded")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    try:
        print_prediction("output.wav")
    except Exception as e:
        print(e)

    # if flag=1:
    #     break

print("Recording ended")














































# import sounddevice as sd
# from scipy.io.wavfile import write

# fs = 44100  # Sample rate
# seconds = 3  # Duration of recording

# myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
# sd.wait()  # Wait until recording is finished
# write('output.wav', fs, myrecording)


