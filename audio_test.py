import pickle, librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_model(model_name):
    return pickle.load(open(model_name, 'rb'))


# Not yet finished
def predict_chord(files, model):
    # For loop through test files
    for file in files:
        # load file
        audio, sample_rate = librosa.load(file) 
        # Get features
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        # Scaled features
        mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)

        # print(mfccs_scaled_features)
        mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
        print(mfccs_scaled_features)
        print(mfccs_scaled_features.shape)

        # Get predicted label
        predicted_label=model.predict(mfccs_scaled_features) 
        predicted_label=np.argmax(predicted_label,axis=1)

        print("File: ", file)
        print("predicted label is:",predicted_label)
        # Get predicted class name
        prediction_class = label_encoder.inverse_transform(predicted_label)
        print("predicted class is:",prediction_class)



test_files = ["chords/Bdim_RockGB_JO_4.wav", "chords/Bb_Electric1_LInda_2.wav", 
              "chords/C_Classic_Jo_1.wav", "chords/chord.wav", 
              "chords/G_AcusticVince_JO_1.wav"]
