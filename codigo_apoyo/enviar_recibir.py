##enviar

import numpy as np
import sounddevice as sd

# Parámetros de la transmisión
SAMPLE_RATE = 44100      # Frecuencia de muestreo (Hz)
CARRIER_FREQ = 17000     # Frecuencia portadora (17 kHz - casi inaudible)
DURATION = 5             # Duración del mensaje en segundos
INDEX_MOD = 0.7          # Índice de modulación AM (m < 1 para evitar sobremodulación)

print("Graba tu mensaje...")
# 1. Grabación del audio original (Señal moduladora)
audio_data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
sd.wait()
audio_signal = audio_data.flatten()

# Normalización del audio grabado entre -1 y 1
if np.max(np.abs(audio_signal)) > 0:
    audio_signal = audio_signal / np.max(np.abs(audio_signal))

# 2. Generación de la señal Portadora
t = np.linspace(0, DURATION, len(audio_signal), endpoint=False)
carrier = np.cos(2 * np.pi * CARRIER_FREQ * t)

# 3. Modulación AM (Doble Banda Lateral con Portadora: s(t) = (1 + m * m(t)) * c(t))
am_signal = (1.0 + INDEX_MOD * audio_signal) * carrier

# Normalización final para evitar distorsión en la salida del altavoz
am_signal = am_signal / np.max(np.abs(am_signal))

print(f"Transmitiendo señal AM a {CARRIER_FREQ} Hz por el altavoz...")
# 4. Reproducción de la señal modulada
sd.play(am_signal, samplerate=SAMPLE_RATE)
sd.wait()
print("Transmisión finalizada.")



##Recibir
import numpy as np
import sounddevice as sd
from scipy.signal import butter, filtfilt, hilbert

# Parámetros (deben coincidir con el emisor)
SAMPLE_RATE = 44100
CARRIER_FREQ = 17000
DURATION = 7             # Se graba un poco más de tiempo para capturar la transmisión completa

print("Receptor escuchando el aire...")
# 1. Grabación del audio ambiental desde el micrófono
rec_data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
sd.wait()
received_signal = rec_data.flatten()

# 2. Filtro Paso Banda centrado en la frecuencia portadora (Aísla la señal AM del ruido)
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# Sintonización: Filtramos solo la banda de interés (16 kHz a 18 kHz)
filtered_signal = bandpass_filter(received_signal, CARRIER_FREQ - 1000, CARRIER_FREQ + 1000, SAMPLE_RATE)

# 3. Demodulación de Envolvente usando la Transformada de Hilbert
analytic_signal = hilbert(filtered_signal)
amplitude_envelope = np.abs(analytic_signal)

# 4. Eliminación del componente DC (offset)
demodulated_signal = amplitude_envelope - np.mean(amplitude_envelope)

# Normalización del audio recuperado
if np.max(np.abs(demodulated_signal)) > 0:
    demodulated_signal = demodulated_signal / np.max(np.abs(demodulated_signal))

print("Reproduciendo el audio demodulado...")
# 5. Reproducción del mensaje original recuperado
sd.play(demodulated_signal, samplerate=SAMPLE_RATE)
sd.wait()
print("Recepción completada.")