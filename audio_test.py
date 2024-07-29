from keras.models import load_model
import librosa
import numpy as np
import tkinter as tk
from tkinter import filedialog
from sklearn.preprocessing import LabelEncoder
import pygame

def predict_chord(file, model, label_encoder):
    audio, sample_rate = librosa.load(file)
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)

    predicted_label = model.predict(mfccs_scaled_features)
    predicted_label = np.argmax(predicted_label, axis=1)

    prediction_class = label_encoder.inverse_transform(predicted_label)
    return prediction_class[0]

def upload_and_predict(model, label_encoder):
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
    if file_path:
        prediction = predict_chord(file_path, model, label_encoder)
        print(f"File: {file_path}")
        print(f"Predicted chord: {prediction}")
        return prediction

pygame.init()
screen = pygame.display.set_mode((640, 480))
pygame.display.set_caption("Chord Detection")

model = load_model("model.h5")
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('classes.npy')

# Main loop
running = True
prediction_text = ""
font = pygame.font.Font(None, 36)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            prediction_text = upload_and_predict(model, label_encoder)

    screen.fill((255, 255, 255)) 
    text_surface = font.render("Click to upload an audio file", True, (0, 0, 0))
    screen.blit(text_surface, (100, 200))

    if prediction_text:
        prediction_surface = font.render(f"Predicted Chord: {prediction_text}", True, (0, 0, 0))
        screen.blit(prediction_surface, (100, 250))

    pygame.display.flip()

pygame.quit()
