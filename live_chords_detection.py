import pygame
import numpy as np
import pyaudio
import librosa
import threading
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import sys

# audio staream params
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22050  # sampling rate
CHUNK = 2048  # chunk = audio_sample/frame
THRESHOLD = 0.4

# rms func for audio data chunk
def calculate_RMS(audio_data):
    return np.sqrt(np.mean(np.square(audio_data)))

# Audio -> Prediction Func
def audio_processing(stream):
    global prediction, rms
    while True:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            # Conv byte -> np array
            audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            # data norm
            audio_data /= np.max(np.abs(audio_data)) + 1e-6
            
            rms = calculate_RMS(audio_data)
            print(rms)

            if rms > THRESHOLD:
                # extract MFCC features
                mfccs = librosa.feature.mfcc(y=audio_data, sr=RATE, n_mfcc=40)
                mfccs_scaled = np.mean(mfccs.T, axis=0)
                mfccs_scaled = mfccs_scaled.reshape(1, -1)

                # pred
                prediction_probs = model.predict(mfccs_scaled)
                predicted_label = np.argmax(prediction_probs, axis=1)
                predicted_chord = label_encoder.inverse_transform(predicted_label)[0]

                # Update the shared prediction variable
                prediction = predicted_chord

        except Exception as e:
            print(f"Error in audio processing thread: {e}")
            prediction = "Error"


rms = 0.1 # <-- init so it has value
# model & class load
model = load_model("model.h5")
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('classes_1.npy', allow_pickle=True)

# pygame init
pygame.init()
screen = pygame.display.set_mode((640, 480))
pygame.display.set_caption("Live Chord Detection")
font = pygame.font.Font(None, 48)
clock = pygame.time.Clock()

prediction = "Listening..."

# pyaudio init
p = pyaudio.PyAudio()

try:
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
except Exception as e:
    print(f"Could not open microphone: {e}")
    sys.exit(-1)

# audio proc thread
audio_thread = threading.Thread(target=audio_processing, args=(stream,), daemon=True)
audio_thread.start()

# main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # cls
    screen.fill((255, 255, 255))

    # Render heading
    instruction_text = font.render("Live Chord Detection", True, (0, 0, 0))
    screen.blit(instruction_text, (160, 150))

    # Render prediction
    pred_text = f"Chord: {prediction}"
    prediction_text = font.render(pred_text, True, (0, 128, 0))
    rms_text = font.render(f"RMS : {rms}", True, (0, 128, 0))
    print(pred_text)
    screen.blit(prediction_text, (150, 250))
    screen.blit(rms_text, (150,280))

    # Update the display
    pygame.display.flip()
    clock.tick(30)  # Limit to 30 FPS

# cleanup after termination
stream.stop_stream()
stream.close()
p.terminate()
pygame.quit()
sys.exit()

