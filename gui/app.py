"""Ventana principal de RETICO."""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from core.audio_io import Audio
from core.pipeline import Digitized, Params, ParamsError, digitize
from core.quantization import QuantizationError

from .tab_captura import TabCaptura
from .tab_digitalizacion import TabDigitalizacion
from .tab_espectro import TabEspectro
from .tab_nyquist import TabNyquist
from .tab_enviar import TabEnviar


class App(tk.Tk):
    """Ventana con barra de parámetros globales y cuatro pestañas."""

    def __init__(self) -> None:
        super().__init__()
        self.title("RETICO — Digitalización de audio")
        self.geometry("1320x900")
        self.minsize(1100, 760)

        self.audio: Audio | None = None
        self._audio_serial = 0
        self._digitized: Digitized | None = None
        self._digitized_key: tuple | None = None
        self.status_label: ttk.Label | None = None

        self._build_vars()
        self._build_style()
        self._build_layout()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # -- variables ---------------------------------------------------------
    def _build_vars(self) -> None:
        self.fs_var = tk.StringVar(value="16000")
        self.bits_var = tk.IntVar(value=8)
        self.vref_var = tk.DoubleVar(value=1.0)
        self.window_var = tk.StringVar(value="Hann")
        self.antialias_var = tk.BooleanVar(value=True)
        self.normalize_var = tk.BooleanVar(value=True)

        self.status_var = tk.StringVar(value="Listo. Grabe desde el micrófono o cargue un WAV.")
        self.audio_var = tk.StringVar(value="Audio: sin cargar.")
        self.params_var = tk.StringVar(value="")

    # -- estilo ------------------------------------------------------------
    def _build_style(self) -> None:
        style = ttk.Style(self)
        for theme in ("vista", "clam", "default"):
            if theme in style.theme_names():
                style.theme_use(theme)
                break
        style.configure("Title.TLabel", font=("Segoe UI", 17, "bold"))
        style.configure("Subtitle.TLabel", font=("Segoe UI", 9), foreground="#555555")
        style.configure("Info.TLabel", font=("Consolas", 9))
        style.configure("Metric.TLabel", font=("Segoe UI", 10, "bold"))
        style.configure("Status.TLabel", font=("Segoe UI", 9), padding=(8, 4))
        style.configure("TLabelframe.Label", font=("Segoe UI", 9, "bold"))

    # -- construcción ------------------------------------------------------
    def _build_layout(self) -> None:
        header = ttk.Frame(self, padding=(12, 8, 12, 2))
        header.pack(side="top", fill="x")
        ttk.Label(header, text="RETICO — Digitalización de audio", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Grabación · muestreo · cuantización · FFT · Nyquist, aliasing y reconstrucción",
            style="Subtitle.TLabel",
        ).pack(anchor="w")

        params = ttk.LabelFrame(self, text="Parámetros globales", padding=(10, 4, 10, 6))
        params.pack(side="top", fill="x", padx=12, pady=(2, 4))

        ttk.Label(params, text="fs objetivo (Hz)").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            params, textvariable=self.fs_var, width=9,
            values=["8000", "11025", "16000", "22050", "32000", "44100", "48000"],
        ).grid(row=1, column=0, padx=(0, 10), sticky="w")

        ttk.Label(params, text="Bits del ADC").grid(row=0, column=1, sticky="w")
        ttk.Spinbox(params, from_=2, to=24, textvariable=self.bits_var,
                    width=5).grid(row=1, column=1, padx=(0, 10), sticky="w")

        ttk.Label(params, text="Vref (V)").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(params, from_=0.1, to=10.0, increment=0.1,
                    textvariable=self.vref_var, width=6).grid(
            row=1, column=2, padx=(0, 10), sticky="w")

        ttk.Label(params, text="Ventana FFT").grid(row=0, column=3, sticky="w")
        ttk.Combobox(
            params, textvariable=self.window_var, width=12, state="readonly",
            values=["Hann", "Hamming", "Blackman", "Rectangular"],
        ).grid(row=1, column=3, padx=(0, 12), sticky="w")

        ttk.Checkbutton(params, text="Filtro antialiasing",
                        variable=self.antialias_var).grid(row=1, column=4, padx=(0, 12), sticky="w")
        ttk.Checkbutton(params, text="Normalizar pico",
                        variable=self.normalize_var).grid(row=1, column=5, padx=(0, 12), sticky="w")
        ttk.Button(params, text="Actualizar", command=self.apply_params).grid(
            row=1, column=6, padx=(0, 4), sticky="w")

        ttk.Label(params, textvariable=self.audio_var).grid(
            row=2, column=0, columnspan=7, sticky="w", pady=(6, 0))
        ttk.Label(params, textvariable=self.params_var, style="Subtitle.TLabel").grid(
            row=3, column=0, columnspan=7, sticky="w")

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(side="top", fill="both", expand=True, padx=12, pady=(2, 4))

        self.tab_captura = TabCaptura(self.notebook, self)
        self.tab_digitalizacion = TabDigitalizacion(self.notebook, self)
        self.tab_espectro = TabEspectro(self.notebook, self)
        self.tab_nyquist = TabNyquist(self.notebook, self, mode="aliasing")
        self.tab_recon = TabNyquist(self.notebook, self, mode="recon")
        self.tab_enviar = TabEnviar(self.notebook, self)
        self.tabs = [self.tab_captura, self.tab_digitalizacion,
                     self.tab_espectro, self.tab_nyquist, self.tab_recon,
                     self.tab_enviar]

        self.notebook.add(self.tab_captura, text="1. Captura")
        self.notebook.add(self.tab_digitalizacion, text="2. Digitalización y cuantización")
        self.notebook.add(self.tab_espectro, text="3. FFT y espectro")
        self.notebook.add(self.tab_nyquist, text="4. Nyquist y aliasing")
        self.notebook.add(self.tab_recon, text="5. Reconstrucción")
        self.notebook.add(self.tab_enviar, text="6. Enviar señal")
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        status = ttk.Frame(self)
        status.pack(side="bottom", fill="x")
        ttk.Separator(status, orient="horizontal").pack(fill="x")
        self.status_label = ttk.Label(status, textvariable=self.status_var,
                                      style="Status.TLabel")
        self.status_label.pack(side="left", fill="x", expand=True)

        self._update_params_label()
        for var in (self.fs_var, self.bits_var, self.vref_var, self.window_var,
                    self.antialias_var, self.normalize_var):
            var.trace_add("write", self._on_params_edited)
        self.bind("<Return>", lambda _event: self.apply_params())

    # -- estado ------------------------------------------------------------
    def set_status(self, text: str, error: bool = False) -> None:
        self.status_var.set(text)
        if self.status_label is not None:
            self.status_label.configure(foreground="#b00020" if error else "#222222")

    def _on_params_edited(self, *_args) -> None:
        self._digitized = None
        self._digitized_key = None
        for tab in self.tabs:
            tab.mark_dirty()
        try:
            self._update_params_label(self.current_params())
        except ParamsError:
            self.params_var.set("Parámetros incompletos.")
        self.set_status("Parámetros modificados. Pulse «Actualizar» o cambie de pestaña.")

    def current_params(self) -> Params:
        try:
            fs = int(round(float(self.fs_var.get().replace(",", "."))))
        except ValueError as exc:
            raise ParamsError("La fs objetivo no es un número válido.") from exc
        try:
            bits = int(self.bits_var.get())
        except (tk.TclError, ValueError) as exc:
            raise ParamsError("Los bits del ADC no son un número válido.") from exc
        try:
            vref = float(self.vref_var.get())
        except (tk.TclError, ValueError) as exc:
            raise ParamsError("Vref no es un número válido.") from exc
        p = Params(fs_target=fs, bits=bits, vref=vref, window=self.window_var.get(),
                   antialias=bool(self.antialias_var.get()),
                   normalize=bool(self.normalize_var.get()))
        return p.validate()

    def get_digitized(self, force: bool = False) -> Digitized | None:
        if self.audio is None:
            return None
        try:
            p = self.current_params()
        except ParamsError as exc:
            self.set_status(str(exc), error=True)
            return None
        key = (self._audio_serial, p.key())
        if force or self._digitized is None or key != self._digitized_key:
            try:
                self._digitized = digitize(self.audio, p)
                self._digitized_key = key
            except (ParamsError, QuantizationError, ValueError) as exc:
                self.set_status(f"Error de procesamiento: {exc}", error=True)
                return None
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror("Error", f"No se pudo procesar el audio:\n{exc}")
                return None
        return self._digitized

    def set_audio(self, audio: Audio) -> None:
        self.audio = audio
        self._audio_serial += 1
        self._digitized = None
        self._digitized_key = None
        bits = "float" if audio.is_float else f"{audio.bits} bits"
        self.audio_var.set(
            f"Audio: {audio.name} · {audio.duration:.3f} s · {audio.fs} Hz · "
            f"{audio.channels} canal(es) · {bits} · {audio.source}")
        self.set_status(f"Audio cargado: {audio.name} ({audio.duration:.2f} s).")
        for tab in self.tabs:
            tab.mark_dirty()
        current = self._current_tab()
        if current is not None:
            current.on_show()

    def apply_params(self) -> None:
        try:
            p = self.current_params()
        except ParamsError as exc:
            messagebox.showerror("Parámetros", str(exc))
            self.set_status(str(exc), error=True)
            return
        self._digitized = None
        self._digitized_key = None
        self._update_params_label(p)
        self.set_status("Parámetros actualizados.")
        current = self._current_tab()
        if current is not None:
            current.mark_dirty()
            current.on_show()

    def _update_params_label(self, p: Params | None = None) -> None:
        if p is None:
            try:
                p = self.current_params()
            except ParamsError:
                self.params_var.set("Parámetros incompletos.")
                return
        aa = "antialias ON" if p.antialias else "antialias OFF"
        norm = "normalizado" if p.normalize else "sin normalizar"
        self.params_var.set(
            f"fs = {p.fs_target} Hz · {p.bits} bits · Vref = ±{p.vref:g} V · "
            f"ventana {p.window} · {aa} · {norm}")

    def _current_tab(self):
        try:
            name = self.notebook.select()
            if not name:
                return None
            widget = self.nametowidget(name)
            return widget if hasattr(widget, "on_show") else None
        except (tk.TclError, KeyError):
            return None

    def _on_tab_changed(self, _event=None) -> None:
        tab = self._current_tab()
        if tab is not None:
            tab.on_show()

    def _on_close(self) -> None:
        for tab in self.tabs:
            try:
                tab.shutdown()
            except Exception:  # noqa: BLE001
                pass
        self.destroy()


def main() -> None:
    App().mainloop()


if __name__ == "__main__":
    main()
