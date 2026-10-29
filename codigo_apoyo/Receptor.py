import numpy as np
import sounddevice as sd
from scipy.signal import butter, filtfilt

# Parámetros (Obligatorio: deben ser idénticos al emisor)
SAMPLE_RATE = 44100
CARRIER_FREQ = 17000
DURATION = 7             # Grabamos más tiempo para asegurar capturar todo el mensaje
CUTOFF_FREQ = 4000       # Frecuencia máxima de la voz humana para el filtro pasa-bajas

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Filtro para sintonizar solo la frecuencia portadora y aislarla del ruido ambiente"""
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data)

def lowpass_filter(data, cutoff, fs, order=4):
    """Filtro Pasa-Bajas: El colador que destruye la portadora y deja la envolvente"""
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='low')
    return filtfilt(b, a, data)

print("🎧 Receptor escuchando el aire...")
# PASO 1: Grabación del aire a través del micrófono
rec_data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
sd.wait()
received_signal = rec_data.flatten()

print("⚙️ Demodulando...")
# PASO 2: Sintonización (Aislar la frecuencia de 17 kHz de los ruidos de la habitación)
filtered_signal = bandpass_filter(received_signal, CARRIER_FREQ - 2000, CARRIER_FREQ + 2000, SAMPLE_RATE)

# PASO 3: Rectificación (Valor Absoluto)
# Volteamos la parte negativa de la señal hacia arriba para que no se anule al filtrar
rectified_signal = np.abs(filtered_signal)

# PASO 4: Detector de Envolvente (Filtro Pasa-Bajas)
# Suavizamos las subidas y bajadas abruptas de la señal rectificada para recuperar la onda lenta
envelop_signal = lowpass_filter(rectified_signal, CUTOFF_FREQ, SAMPLE_RATE)

# PASO 5: Supresión del Offset (Centrar en cero)
# Le restamos el promedio DC para que la onda de audio vuelva a oscilar entre negativo y positivo
demodulated_signal = envelop_signal - np.mean(envelop_signal)

# Normalizar para escuchar a buen volumen
if np.max(np.abs(demodulated_signal)) > 0:
    demodulated_signal = demodulated_signal / np.max(np.abs(demodulated_signal))

print("▶️ Reproduciendo el audio original recuperado...")
# PASO 6: Reproducir el resultado final
sd.play(demodulated_signal, samplerate=SAMPLE_RATE)
sd.wait()
print("✅ Recepción completada.")