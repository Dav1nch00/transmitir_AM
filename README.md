# RETICO — Digitalización de audio y transmisión AM

Aplicación de escritorio en Python (Tkinter + Matplotlib) para grabar audio,
digitalizarlo, analizarlo con FFT y estudiar muestreo, cuantización, Nyquist,
aliasing y reconstrucción. Incluye además la transmisión y recepción acústica
de una señal modulada en AM (DSB-FC).

El flujo completo es:

```
micrófono / WAV  →  x(t)  →  muestreo fs  →  cuantización N bits  →  tramas PCM
                                      │
                                      ├── FFT y espectrograma
                                      ├── Nyquist y aliasing
                                      ├── reconstrucción DAC / sinc
                                      └── AM: (1 + m·x(t))·cos(2π·fc·t)
                                             └── altavoz ── aire ── micrófono ── demodulación
```

## Requisitos

- Python 3.10 o superior.
- `numpy`, `scipy`, `matplotlib`, `PyAudio` y Tkinter.
- En Windows, Tkinter viene con Python. Para instalar el resto:
  ```bash
  pip install -r requirements.txt
  ```
- En Debian/Ubuntu, además:
  ```bash
  sudo apt install python3-tk python3-venv portaudio19-dev
  python3 -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  ```

## Ejecución

```bash
python main.py
```

La primera vez, el programa busca los micrófonos disponibles. El dispositivo
predeterminado del equipo se selecciona automáticamente. Si un dispositivo no
acepta la frecuencia elegida, pruebe con **44100 Hz**.

## Estructura

```
transmitir_AM/
├── main.py                  punto de entrada
├── requirements.txt
├── README.md
├── core/                    lógica DSP y AM, sin depender de la interfaz
│   ├── audio_io.py          micrófono, lectura/escritura WAV y reproducción
│   ├── dsp.py               FFT, ventanas, remuestreo, aliasing, sinc y STFT
│   ├── quantization.py      ADC/DAC, Δ, códigos binarios, error y SQNR
│   ├── pipeline.py          cadena audio → x[n] → cuantización
│   └── am.py                modulación y demodulación AM (DSB-FC)
├── gui/
│   ├── app.py               ventana, parámetros globales y pestañas
│   ├── widgets.py           figura incrustada, barra de guardado y CSV
│   ├── tab_captura.py       pestaña 1
│   ├── tab_digitalizacion.py pestaña 2
│   ├── tab_espectro.py      pestaña 3
│   ├── tab_nyquist.py       pestañas 4 y 5 (aliasing y reconstrucción)
│   ├── tab_enviar.py        pestaña 6 (transmisión AM)
│   └── tab_recibir.py       pestaña 7 (recepción AM)
├── tests/                   pruebas del núcleo y de AM
├── salidas/                 WAV, CSV y figuras exportadas
└── referencia/              scripts originales de apoyo (Emisor.py, Receptor.py, enviar_recibir.py)
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

### 5. Reconstrucción

- Comparación entre retención de orden cero (escalera del DAC) e interpolación
  sinc.
- Error entre ambas y reproducción de la señal original, cuantizada y
  reconstruida.

### 6. Transmitir AM

- Fuente de la señal: la salida digitalizada del DAC (x[n] normalizado) o el
  audio original.
- Portadora ajustable (predeterminada 17000 Hz) e índice de modulación m.
- La banda base se normaliza y remuestrea a 44100 Hz con antialiasing, se
  modula y se reproduce por el altavoz.
- Gráficas de la banda base y de la señal modulada en una ventana de 50 ms.

### 7. Recibir AM

- Captura de la señal transmitida desde el micrófono (7 s a 44100 Hz por
  defecto, configurable entre 3 y 15 s).
- Demodulación: filtro pasa-bandas alrededor de la portadora (fc ± 2000 Hz),
  rectificación y filtro pasa-bajas de 4000 Hz para extraer la envolvente.
- El audio demodulado alimenta el estado compartido: se puede reproducir,
  analizar con FFT, digitalizar de nuevo y estudiar su Nyquist/reconstrucción.
- Medidor de nivel y temporizador durante la recepción.

### 8. Exportación

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
escritura de WAV y la cadena completa de digitalización. Las pruebas de AM
(`tests/test_am.py`) cubren la frecuencia de la portadora y las bandas
laterales, el remuestreo de banda base, la recuperación de un tono con
envolvente y con Hilbert, el aplanado de la envolvente con solo portadora y
los filtros pasa-bandas y pasa-bajas.

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

## Referencias

- Proakis & Manolakis, *Digital Signal Processing*, capítulos de muestreo,  
  cuantización y modulación de amplitud.
- A. V. Oppenheim & R. W. Schafer, *Discrete-Time Signal Processing*.
- `referencia/`: scripts originales de apoyo (Emisor.py, Receptor.py, enviar_recibir.py).