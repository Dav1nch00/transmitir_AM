# RETICO — Digitalización de audio

Aplicación de escritorio en Python (Tkinter + Matplotlib) para grabar audio,
digitalizarlo, analizarlo con FFT y estudiar muestreo, cuantización, Nyquist,
aliasing y reconstrucción.

El flujo completo es:

```
micrófono / WAV  →  x(t)  →  muestreo fs  →  cuantización N bits  →  tramas PCM
                                      │
                                      ├── FFT y espectrograma
                                      ├── Nyquist y aliasing
                                      └── reconstrucción DAC / sinc
```

## Requisitos

- Python 3.10 o superior.
- `numpy`, `scipy`, `matplotlib`, `PyAudio` y Tkinter.

En Windows, Tkinter viene con Python. Para instalar el resto:

```bash
pip install -r requirements.txt
```

En Debian/Ubuntu, además:

```bash
sudo apt install python3-tk python3-venv portaudio19-dev
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecución

```bash
cd retico
python main.py
```

La primera vez, el programa busca los micrófonos disponibles. El dispositivo
predeterminado del equipo se selecciona automáticamente. Si un dispositivo no
acepta la frecuencia elegida, pruebe con **44100 Hz**.

## Estructura

```
retico/
├── main.py                  punto de entrada
├── requirements.txt
├── README.md
├── core/                    lógica DSP, sin depender de la interfaz
│   ├── audio_io.py          micrófono, lectura/escritura WAV y reproducción
│   ├── dsp.py               FFT, ventanas, remuestreo, aliasing, sinc y STFT
│   ├── quantization.py      ADC/DAC, Δ, códigos binarios, error y SQNR
│   └── pipeline.py          cadena audio → x[n] → cuantización
├── gui/
│   ├── app.py               ventana, parámetros globales y pestañas
│   ├── widgets.py           figura incrustada, barra de guardado y CSV
│   ├── tab_captura.py       pestaña 1
│   ├── tab_digitalizacion.py pestaña 2
│   ├── tab_espectro.py      pestaña 3
│   └── tab_nyquist.py       pestaña 4
├── tests/test_core.py       pruebas del núcleo
└── salidas/                 WAV, CSV y figuras exportadas
```

## Parámetros globales

| Parámetro | Significado |
|---|---|
| **fs objetivo** | Frecuencia a la que se muestrea x(t). Se aplica a todas las pestañas. |
| **Bits del ADC** | Resolución del cuantizador (2 a 24 bits). |
| **Vref** | Semirrango de entrada del ADC bipolar: de −Vref a +Vref. |
| **Ventana FFT** | Hann, Hamming, Blackman o rectangular. |
| **Filtro antialiasing** | Activa el filtro paso bajo previo al remuestreo. Al desactivarlo se ve el plegado real. |
| **Normalizar pico** | Lleva el pico al 95 % de la escala completa antes de cuantizar. |

## Pestañas

### 1. Captura

- Lista y selección del micrófono.
- Grabación de 1 a 60 s a 8, 16, 22.05, 32, 44.1 o 48 kHz.
- Medidor de nivel y temporizador.
- Reproducción del audio cargado o grabado.
- Carga de WAV PCM de 8, 16, 24 o 32 bits, mono o estéreo (se promedia a mono).
- Exportación a WAV PCM de 16 bits en `salidas/`.

### 2. Digitalización y cuantización

- Compara x(t) continua con x[n] y con la escalera del DAC ideal.
- Muestra el error de cuantización y la cota ±Δ/2.
- Tabla con `n`, tiempo, `x(n)`, tensión, código con signo, código binario,
  `V_DAC` y error en mV.
- Métricas: Δ, niveles, error máximo y RMS, muestras saturadas, SQNR medida,
  SQNR esperada, SQNR de senoide a plena escala, ENOB y bitrate.
- Exportación de la tabla a CSV y tabla aparte con los niveles del ADC.

### 3. FFT y espectro

- Espectro unilateral de amplitud en V o dBV, para la señal original y para x[n].
- Frecuencia del pico, ancho de banda al 99 % de la energía, centroide espectral
  y resolución `Δf = fs/N`.
- Espectrograma (STFT) de la señal digitalizada.

### 4. Nyquist, aliasing y reconstrucción

Dos vistas:

**Muestreo y aliasing**

- Espectro original con los límites de Nyquist de varias fs.
- Espectros de la señal muestreada a 8, 16, 22.05 y 44.1 kHz.
- Tabla con fs, Nyquist, `f_max` al 99 %, energía que se plegaría y veredicto.
- Demostración de un tono: por ejemplo, 7000 Hz muestreado a 8000 Hz aparece
  como 1000 Hz.

**Reconstrucción**

- Comparación entre retención de orden cero (escalera del DAC) e interpolación
  sinc.
- Error entre ambas y reproducción de la señal original, cuantizada y
  reconstruida.

## Fórmulas implementadas

### Muestreo y Nyquist

```
x[n] = x(n·Ts),        Ts = 1/fs
fs ≥ 2·f_max           (Nyquist-Shannon)
f_alias = |f − k·fs|   (k entero)
```

`f_max` se estima como la frecuencia por debajo de la cual se acumula el 99 %
de la energía del espectro unilateral. No es una verdad absoluta: es una
convención útil para audio real.

### FFT

```
|X(f)| = 2·|FFT{x[n]·w[n]}| / Σw[n]
Δf = fs/N
```

El factor 2 reconstruye la amplitud de la sinusoide en el espectro unilateral;
DC y Nyquist no se duplican.

### Cuantización

```
Δ = 2·Vref / 2^N
v(n) = x(n)·Vref
q(n) = clip(round(v/Δ), −2^(N−1), 2^(N−1)−1)
código ADC = q(n) + 2^(N−1)
V_DAC(n) = q(n)·Δ
−Δ/2 ≤ e(n) = V_DAC(n) − v(n) ≤ Δ/2
```

### Relación señal-ruido de cuantización

Para una senoide a plena escala:

```
SQNR = 6.02·N + 1.76 dB
```

Para una señal cualquiera, el programa calcula también:

```
SQNR_medida = 10·log10( P_señal / P_error )
SQNR_esperada = 10·log10( P_señal / (Δ²/12) )
ENOB = (SQNR_medida − 1.76) / 6.02
```

### Reconstrucción

```
x(t) = Σ x[n]·sinc((t − n·Ts)/Ts)
```

La implementación trunca el sinc a 65 muestras (32 a cada lado) y usa
retención de orden cero para representar el DAC.

## Experimentos para la sustentación

1. **Ruido de cuantización.** Grabe la misma frase con 4, 8 y 16 bits. Observe
   la escalera, el error y la SQNR. A 4 bits la voz se entiende pero suena
   granulada; a 16 bits el error es inaudible.
2. **Nyquist y ancho de banda.** Cargue una canción y compare fs = 8000 Hz con
   fs = 44100 Hz. La pestaña 4 indica si se cumple `fs ≥ 2·f_max`.
3. **Aliasing.** En la pestaña 4, active el filtro antialiasing y luego
   desactívelo. Con un tono de 7000 Hz a fs = 8000 Hz, la frecuencia aparente
   es 1000 Hz. El espectro muestreado lo muestra sin ambigüedad.
4. **Reconstrucción.** Compare la escalera del DAC con el sinc. A menor fs, la
   diferencia entre ambas es más grande y la escalera se vuelve audible.
5. **Resolución frecuencial.** Use ventana rectangular y luego Hann: la fuga
   espectral disminuye, pero el lóbulo principal se ensancha.

## Exportación

- **WAV:** pestaña 1, botón «Guardar WAV» (PCM 16 bits).
- **CSV de muestras:** pestaña 2, hasta 100 000 filas.
- **Figuras:** botón de guardado de la barra de Matplotlib en cada pestaña.
- Los botones de WAV y CSV proponen la carpeta `salidas/` con marca de tiempo;
  las figuras usan el diálogo de Matplotlib.

## Pruebas

```bash
python -m unittest discover -s tests -v
```

Las pruebas cubren cuantización, cota del error, SQNR, FFT, ancho de banda,
frecuencia de aliasing, filtro antialiasing, interpolación sinc, lectura y
escritura de WAV y la cadena completa de digitalización.

## Limitaciones y notas

- La grabación está limitada a 60 s para no consumir memoria innecesariamente.
- La reproducción usa el dispositivo de salida predeterminado del sistema.
- En la pestaña 4, la comparación de fs usa como máximo los primeros 10 s del
  audio para mantener la interfaz fluida.
- El filtro antialiasing es un polifásico con ventana Kaiser de `scipy`; no es
  un Butterworth, pero su misión es la misma: atenuar lo que se pliega.
- El WAV exportado es de 16 bits; la cuantización interna puede usar cualquier
  resolución entre 2 y 24 bits.
- La aplicación no reemplaza un analizador de espectro calibrado: trabaja con
  la escala del archivo y con Vref como referencia.
