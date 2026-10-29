import numpy as np
import sounddevice as sd

# Parámetros de la transmisión
SAMPLE_RATE = 44100      # Frecuencia de muestreo estándar
CARRIER_FREQ = 17000     # Frecuencia de la portadora (17 kHz - agudo extremo)
DURATION = 5             # Duración de la grabación en segundos
INDEX_MOD = 0.7          # Índice de modulación (m)

print("🎙️ Grabando tu mensaje... Habla ahora.")
# PASO 1: Grabación de la señal de banda base (el audio original)
audio_data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
sd.wait()
audio_signal = audio_data.flatten()

# Normalización del audio entre -1 y 1
if np.max(np.abs(audio_signal)) > 0:
    audio_signal = audio_signal / np.max(np.abs(audio_signal))

print("⚙️ Modulando la señal...")
# PASO 2: Generación de la portadora de alta frecuencia
t = np.linspace(0, DURATION, len(audio_signal), endpoint=False)
carrier = np.cos(2 * np.pi * CARRIER_FREQ * t)

# PASO 3: El problema del cruce por cero y la mezcla (Modulación AM)
# - El "1.0" es el DC OFFSET que sube la señal por encima de cero.
# - El "*" es la MULTIPLICACIÓN que aplica la envolvente a la portadora.
am_signal = (1.0 + INDEX_MOD * audio_signal) * carrier

# Normalizar la señal modulada para evitar que el altavoz distorsione
am_signal = am_signal / np.max(np.abs(am_signal))

print(f"🔊 Transmitiendo ráfaga AM a {CARRIER_FREQ} Hz por los altavoces...")
# PASO 4: Transmisión acústica al aire
sd.play(am_signal, samplerate=SAMPLE_RATE)
sd.wait()
print("✅ Transmisión finalizada.")