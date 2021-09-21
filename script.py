from pandas.core.accessor import DirNamesMixin
from tensorflow import keras
import numpy as np
import librosa
import os

input = open("parameters.txt","r")
modelp = input.readline()[:-1]
num_rows = int(input.readline()[:-1])
num_columns = int(input.readline()[:-1])
num_channels = int(input.readline()[:-1])

a = 'MLP' if modelp == '1' else 'CNN'
print('Classification using',a,'\n')

Classes = ['air_conditioner','car_horn','children_playing','dog_bark','drilling','engine_idling','gun_shot','jackhammer','siren','street_music']
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
        prediction_feature = extract_feature('./test/'+file_name) 

        predicted_vector = model.predict(prediction_feature)
        predicted_class = np.argmax(predicted_vector)
        # predicted_class = le.inverse_transform(predicted_vector) 
        print("The predicted class is:", Classes[predicted_class], '\n')
else:
    max_pad_len = 174

    def extract_features(file_name):
    
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
            
        return mfccs

    def print_prediction(file_name):
        prediction_feature = extract_features('./test/'+file_name) 
        # print(prediction_feature)
        prediction_feature = prediction_feature.reshape(1, num_rows, num_columns, num_channels)

        predicted_vector = model.predict(prediction_feature)
        predicted_class = np.argmax(predicted_vector)
        # predicted_class = le.inverse_transform(predicted_vector) 
        print("The predicted class is:", Classes[predicted_class], '\n')


for d in os.listdir("./test"):
    print('Output of the file ' + d + ' is:')
    print_prediction(d)