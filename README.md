# RETICO — Digitalización de audio y transmisión de señales

Aplicación de escritorio en Python (Tkinter + Matplotlib) para grabar audio,
digitalizarlo, analizarlo con FFT y estudiar muestreo, cuantización, Nyquist,
aliasing y reconstrucción. Incluye además la modulación, transmisión acústica
y recepción/demodulación de la señal con AM, FM o técnicas digitales
(ASK, FSK, PSK, QAM, GFSK y MFSK).

El flujo completo es:

```
micrófono / WAV  →  x(t)  →  muestreo fs  →  cuantización N bits  →  tramas PCM
                                      │
                                      ├── FFT y espectrograma
                                      ├── Nyquist y aliasing
                                      ├── reconstrucción DAC / sinc
                                      └── modulación: AM · FM · ASK · FSK · PSK · QAM · GFSK · MFSK
                                             ├── WAV de la señal modulada
                                             └── altavoz ── aire ── micrófono ── demodulación
                                                                        (AM · FM · ASK · FSK · PSK · QAM · GFSK · MFSK)
                                                                               └── comparación original vs. recuperado
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
│   ├── am.py                modulación y demodulación AM (DSB-FC)
│   ├── modulation.py        FM y modulación/demodulación digital
│   │                        (ASK/FSK/PSK/QAM/GFSK/MFSK)
│   └── compare.py           alineación y métricas original vs. recuperado
├── gui/
│   ├── app.py               ventana, parámetros globales y pestañas
│   ├── widgets.py           figura incrustada, barra de guardado y CSV
│   ├── tab_captura.py       pestaña 1
│   ├── tab_digitalizacion.py pestaña 2
│   ├── tab_espectro.py      pestaña 3
│   ├── tab_nyquist.py       pestañas 4 y 5 (aliasing y reconstrucción)
│   ├── tab_enviar.py        pestaña 6 (modulación y transmisión)
│   ├── tab_recibir.py       pestaña 7 (recepción y demodulación)
│   └── tab_comparar.py      pestaña 8 (comparación de audio)
├── tests/                   pruebas del núcleo, AM, modulación y demodulación
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

- Dos vistas conmutables sin volver a digitalizar:
  - **Digitalización:** compara x(t) continua con x[n] y con la escalera del DAC ideal.
  - **Muestreo Nyquist (NQS):** muestra x(t), las muestras x[n] y el espectro con
    `f_max` al 99 %, `fs`, el límite `fs/2` y el veredicto del criterio de
    Nyquist–Shannon (`fs ≥ 2·f_max`).
- Selector **Bit / Byte** para mostrar la resolución y el flujo PCM en binario o
  en hexadecimal por bytes. El cambio es solo de visualización: no recarga el
  audio ni vuelve a ejecutar la digitalización.
- Muestra el error de cuantización y la cota ±Δ/2.
- Tabla con `n`, tiempo, `x(n)`, tensión, código con signo, código binario o hex,
  `V_DAC` y error en mV.
- Métricas: Δ, niveles, error máximo y RMS, muestras saturadas, SQNR medida,
  SQNR esperada, SQNR de senoide a plena escala, ENOB y bitrate.
- Exportación de la tabla a CSV (en bits o bytes según el selector) y tabla
  aparte con los niveles del ADC.

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

### 6. Modulación y transmisión

- **Técnica seleccionable:** AM, FM, ASK, FSK, PSK, QAM, GFSK o MFSK.
- **Fuente:** la salida digitalizada del DAC (x[n] normalizado) o el audio
  original.
- **Parámetros según la técnica:**
  - AM: portadora e índice de modulación m.
  - FM: portadora y desviación Δf.
  - ASK/PSK/QAM: portadora, baudios y bits por símbolo (M = 2^bps; QAM admite
    2, 4 o 6 bits/símbolo).
  - FSK/MFSK/GFSK: además, desviación Δf; GFSK agrega el producto BT.
  - Técnicas digitales: número de muestras PCM que se convierten en bits.
- La banda base analógica se normaliza y remuestrea a 44100 Hz con antialiasing.
- **Generar modulada** calcula la señal y la grafica (banda base/bits y señal
  modulada).
- Después de generarla hay dos acciones:
  - **Guardar WAV modulada:** exporta la señal modulada en PCM de 16 bits.
  - **Transmitir por el aire:** reproduce la señal modulada por el altavoz
    (con opción de detenerla).

### 7. Recepción y demodulación

El flujo de la pestaña es: **elegir técnica → elegir origen → ajustar
parámetros → Demodular → ver la gráfica y escuchar/guardar el audio**.

1. **Técnica de la señal recibida:** AM, FM, ASK, FSK, PSK, QAM, GFSK o MFSK.
2. **Origen:**
   - **Micrófono (escuchar):** captura la señal del aire (7 s a 44100 Hz por
     defecto, configurable entre 3 y 15 s). Se busca automáticamente el inicio
     de la ráfaga.
   - **Archivo WAV modulado:** carga un WAV generado en la pestaña 6 (o
     cualquier WAV con la señal modulada). Se asume que la señal empieza en la
     muestra 0.
3. **Parámetros de demodulación:**
   - AM: portadora y corte de audio.
   - FM: portadora, desviación Δf y corte de audio.
   - Digitales: portadora, baudios, bits/símbolo, bits PCM, fs del audio y,
     según la técnica, Δf (FSK/MFSK/GFSK) o el producto BT (GFSK).
   - Ganancia en dB y búsqueda del inicio de la ráfaga.
4. **Demodular** procesa la señal recibida.
5. Se muestran dos gráficas (señal modulada recibida y señal demodulada) y se
   habilitan **Reproducir**, **Silencio** y **Guardar audio**.

Detalles de la demodulación:

- AM: filtro pasa-bandas alrededor de la portadora y detección de envolvente.
- FM: filtro pasa-bandas, señal analítica de Hilbert, fase instantánea y
  discriminador de frecuencia.
- Digitales: banco de correladores I/Q (ASK/PSK/QAM) o banco no coherente en
  las M frecuencias posibles (FSK/MFSK/GFSK), decisión de símbolos, conversión
  símbolos → bits → códigos PCM y reconstrucción del audio a la fs original.
- La pestaña adopta automáticamente la técnica y los parámetros de la última
  transmisión generada en la pestaña 6.
- El audio demodulado se puede reproducir y guardar como WAV PCM de 16 bits.
  También pasa a ser el audio compartido de la aplicación: puede analizarse con
  FFT, digitalizarse de nuevo o estudiarse en Nyquist/reconstrucción.
- En GFSK conviene BT ≥ 0.5. En FSK/MFSK la separación entre frecuencias
  (2Δf/(M−1)) debe ser al menos la tasa de símbolos para una decisión fiable.

### 8. Comparación de audio (punto 6)

Compara el audio original con el audio recuperado **sin superponer** las
señales: cada una se dibuja en su propio eje.

- **A · audio original:** última referencia de la transmisión (el audio o la
  señal digitalizada que entró al modulador), el audio actual o un WAV cargado.
  En las técnicas digitales con fuente «Audio original», A es el audio sin
  cuantizar, de modo que la comparación incluye también el error de
  cuantización PCM del transmisor.
- **B · audio recuperado:** última demodulación, el audio actual o un WAV.
- **Alinear automáticamente:** busca el desfase por correlación dentro de
  ±0.5 s (configurable) y recorta al tramo común. **Ajustar ganancia:** aplica
  la escala de mínimos cuadrados para comparar la forma, no el nivel.
- **Vista de tiempo:** A, B y el error `B − A` en tres ejes separados.
- **Vista de espectro:** espectro de A, espectro de B y diferencia de magnitud
  `B − A`, también en ejes separados.
- **Métricas:** desfase en muestras y milisegundos, ganancia aplicada,
  correlación, fidelidad, SNR, RMSE, error máximo, RMS de cada audio y
  porcentaje del tramo comparado.
- **Reproducción separada:** `▶ A`, `▶ B` y `▶ A → B`. La última reproduce
  primero el original, luego una pausa breve y después el recuperado: nunca se
  mezclan.
- **WAV secuencial:** `💾 WAV A → B` guarda esa secuencia (original, pausa,
  recuperado) en un WAV PCM de 16 bits.
- **CSV:** exporta tiempo, A, B y error `B − A` (hasta 100 000 filas).

### 9. Exportación

- **Gráficas:** botón «Guardar gráfica» en la barra inferior de cada pestaña
  (PNG, PDF o SVG en `salidas/`). Guarda la figura completa de esa pestaña.
- **WAV:** pestaña 1, botón «Guardar WAV» (PCM 16 bits); pestaña 6, botón
  «Guardar WAV modulada» (PCM 16 bits de la señal modulada); pestaña 7,
  botón «Guardar audio» (PCM 16 bits del audio demodulado); pestaña 8,
  botón «WAV A → B» (original y recuperado en secuencia, sin mezcla).
- **CSV de muestras:** pestaña 2, hasta 100 000 filas, en binario o en bytes
  hexadecimales según el selector Bit/Byte. Pestaña 8, comparación A/B/error
  también hasta 100 000 filas.
- **Figuras:** además del botón propio, la barra de Matplotlib de cada pestaña
  permite guardar la figura.
- Los botones de WAV, CSV y gráfica proponen la carpeta `salidas/` con marca
  de tiempo.

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
los filtros pasa-bandas y pasa-bajas. Las pruebas de modulación
(`tests/test_modulation.py`) cubren la conversión PCM a bits y la generación
de FM, ASK, FSK, PSK, QAM, GFSK y MFSK. Las pruebas de demodulación
(`tests/test_demodulation.py`) cubren la recuperación de FM, la ronda completa
de bits y de códigos PCM para las seis técnicas digitales, la búsqueda del
inicio de ráfaga y la validación de parámetros. Las pruebas de comparación
(`tests/test_compare.py`) cubren la alineación con desfases positivos y
negativos, el ajuste de ganancia, la inversión de polaridad, el remuestreo, el
ruido, el silencio y la reproducción secuencial A → B.

## Limitaciones y notas

- La grabación está limitada a 60 s para no consumir memoria innecesariamente.
- La reproducción usa el dispositivo de salida predeterminado del sistema.
- En la pestaña 4, la comparación de fs usa como máximo los primeros 10 s del
  audio para mantener la interfaz fluida.
- El filtro antialiasing es un polifásico con ventana Kaiser de `scipy`; no es  
  un Butterworth, pero su misión es la misma: atenuar lo que se pliega.
- El WAV exportado es de 16 bits; la cuantización interna puede usar cualquier  
  resolución entre 2 y 24 bits.
- La recepción acústica funciona con todas las técnicas, pero es una
  demostración: la demodulación digital por micrófono puede tener errores de
  bit por ruido, eco y desincronización. Para una reconstrucción exacta conviene
  cargar el WAV modulado generado por la propia aplicación.
- En GFSK se recomienda BT ≥ 0.5. En FSK/MFSK, si la separación entre
  frecuencias es menor que la tasa de símbolos, la decisión puede fallar;
  aumente Δf o reduzca los baudios.
- La comparación alinea las señales por correlación y recorta al tramo común;
  por eso los filtros de recepción (que descartan unos milisegundos en los
  bordes) aparecen como un pequeño desfase. La comparación no corrige el nivel
  de forma automática salvo que se active «Ajustar ganancia».
- La aplicación no reemplaza un analizador de espectro calibrado: trabaja con  
  la escala del archivo y con Vref como referencia.

## Referencias

- Proakis & Manolakis, *Digital Signal Processing*, capítulos de muestreo,  
  cuantización y modulación de amplitud.
- A. V. Oppenheim & R. W. Schafer, *Discrete-Time Signal Processing*.
- `referencia/`: scripts originales de apoyo (Emisor.py, Receptor.py, enviar_recibir.py).