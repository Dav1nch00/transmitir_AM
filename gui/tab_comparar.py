"""Pestaña 8 (punto 6): comparación entre audio original y recuperado.

La comparación se muestra siempre en ejes separados: nunca se superponen las
dos formas de onda ni los dos espectros.  También permite escuchar los audios
de forma secuencial (A, pausa, B) y exportar la comparación a CSV.
"""
from __future__ import annotations

import math
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from core.audio_io import Audio, AudioError, Player, read_wav, save_wav
from core.compare import ComparisonResult, compare_audio, sequential_signal
from core.dsp import db, display_slice, spectrum

from .widgets import PlotTab, output_dir, save_rows_csv, timestamp

A_TX = "Última transmisión (referencia)"
A_CURRENT = "Audio actual"
B_DEMOD = "Última demodulación"
B_CURRENT = "Audio actual"

MAX_CSV_ROWS = 100_000
MAX_FFT = 65_536


class TabComparar(PlotTab):
    """Compara el audio transmitido con el audio recuperado."""

    def __init__(self, master, app):
        super().__init__(master, app)
        self.a_var = tk.StringVar(value=A_TX)
        self.b_var = tk.StringVar(value=B_DEMOD)
        self.align_var = tk.BooleanVar(value=True)
        self.gain_var = tk.BooleanVar(value=True)
        self.delay_var = tk.StringVar(value="0.5")
        self.view_var = tk.StringVar(value="Tiempo")
        self.info_var = tk.StringVar(value="")
        self.sources_var = tk.StringVar(value="")

        self._a_file: Audio | None = None
        self._a_file_label = ""
        self._b_file: Audio | None = None
        self._b_file_label = ""
        self._a_audio: Audio | None = None
        self._b_audio: Audio | None = None
        self._result: ComparisonResult | None = None

        self._player: Player | None = None
        self._poll_id: str | None = None

        self._build_controls()
        self._update_sources()

    # ------------------------------------------------------------------
    # Interfaz
    # ------------------------------------------------------------------
    def _build_controls(self) -> None:
        c = self.controls

        row1 = ttk.Frame(c)
        row1.pack(fill="x")

        ttk.Label(row1, text="A · audio original").pack(side="left", padx=(0, 4))
        self.a_combo = ttk.Combobox(
            row1, textvariable=self.a_var, state="readonly", width=31,
            values=self._values_a())
        self.a_combo.pack(side="left", padx=(0, 4))
        self.a_combo.bind("<<ComboboxSelected>>", self._on_source_changed)
        ttk.Button(row1, text="📂 WAV A", width=10,
                   command=self._load_a).pack(side="left", padx=(0, 12))

        ttk.Label(row1, text="B · audio recuperado").pack(side="left",
                                                          padx=(0, 4))
        self.b_combo = ttk.Combobox(
            row1, textvariable=self.b_var, state="readonly", width=27,
            values=self._values_b())
        self.b_combo.pack(side="left", padx=(0, 4))
        self.b_combo.bind("<<ComboboxSelected>>", self._on_source_changed)
        ttk.Button(row1, text="📂 WAV B", width=10,
                   command=self._load_b).pack(side="left", padx=(0, 8))

        ttk.Button(row1, text="🔄 Últimas señales", width=17,
                   command=self._use_latest).pack(side="left", padx=(0, 4))

        row2 = ttk.Frame(c)
        row2.pack(fill="x", pady=(4, 0))

        ttk.Checkbutton(row2, text="Alinear automáticamente",
                        variable=self.align_var).pack(side="left", padx=(0, 8))
        ttk.Label(row2, text="Búsqueda ± (s)").pack(side="left", padx=(0, 4))
        ttk.Spinbox(row2, from_=0.0, to=10.0, increment=0.05, width=5,
                    textvariable=self.delay_var).pack(side="left", padx=(0, 10))
        ttk.Checkbutton(row2, text="Ajustar ganancia",
                        variable=self.gain_var).pack(side="left", padx=(0, 12))

        self.compare_btn = ttk.Button(row2, text="📊 Comparar",
                                      command=self._compare)
        self.compare_btn.pack(side="left", padx=(0, 8))

        self.play_a_btn = ttk.Button(row2, text="▶ A", width=5,
                                     command=lambda: self._play("a"),
                                     state="disabled")
        self.play_a_btn.pack(side="left", padx=(0, 4))
        self.play_b_btn = ttk.Button(row2, text="▶ B", width=5,
                                     command=lambda: self._play("b"),
                                     state="disabled")
        self.play_b_btn.pack(side="left", padx=(0, 4))
        self.play_ab_btn = ttk.Button(
            row2, text="▶ A → B", width=7, command=lambda: self._play("ab"),
            state="disabled")
        self.play_ab_btn.pack(side="left", padx=(0, 4))
        self.stop_btn = ttk.Button(row2, text="■ Detener", width=9,
                                   command=self._stop_play, state="disabled")
        self.stop_btn.pack(side="left", padx=(0, 8))

        self.csv_btn = ttk.Button(row2, text="💾 CSV", width=8,
                                  command=self._save_csv, state="disabled")
        self.csv_btn.pack(side="left", padx=(0, 4))

        self.save_ab_btn = ttk.Button(
            row2, text="💾 WAV A → B", width=13,
            command=self._save_sequential, state="disabled")
        self.save_ab_btn.pack(side="left", padx=(0, 4))

        row3 = ttk.Frame(c)
        row3.pack(fill="x", pady=(4, 0))
        ttk.Label(row3, text="Vista").pack(side="left", padx=(0, 4))
        self.view_combo = ttk.Combobox(
            row3, textvariable=self.view_var, state="readonly", width=9,
            values=["Tiempo", "Espectro"])
        self.view_combo.pack(side="left", padx=(0, 12))
        self.view_combo.bind("<<ComboboxSelected>>", self._on_view_changed)
        ttk.Label(row3, textvariable=self.sources_var,
                  style="Subtitle.TLabel").pack(side="left")

        ttk.Label(c, textvariable=self.info_var, style="Info.TLabel",
                  wraplength=1240, justify="left").pack(anchor="w",
                                                        pady=(4, 0))
        ttk.Label(
            c,
            text="Ejes separados, sin superposición · «▶ A → B» reproduce en "
                 "secuencia · «💾 WAV A → B» guarda esa secuencia.",
            style="Subtitle.TLabel", wraplength=1240, justify="left").pack(
            anchor="w", pady=(2, 0))

        self.message("Transmita un audio, demodúlelo y vuelva a esta pestaña: "
                     "la comparación usa la última transmisión y la última "
                     "demodulación.\nTambién puede cargar dos WAV cualesquiera.")

    # ------------------------------------------------------------------
    # Orígenes
    # ------------------------------------------------------------------
    def _values_a(self) -> list[str]:
        values = [A_TX, A_CURRENT]
        if self._a_file is not None:
            values.append(self._a_file_label)
        return values

    def _values_b(self) -> list[str]:
        values = [B_DEMOD, B_CURRENT]
        if self._b_file is not None:
            values.append(self._b_file_label)
        return values

    def _resolve_a(self) -> Audio | None:
        if self._a_file is not None and self.a_var.get() == self._a_file_label:
            return self._a_file
        if self.a_var.get() == A_CURRENT:
            return self.app.audio
        return getattr(self.app, "last_transmission_reference", None)

    def _resolve_b(self) -> Audio | None:
        if self._b_file is not None and self.b_var.get() == self._b_file_label:
            return self._b_file
        if self.b_var.get() == B_CURRENT:
            return self.app.audio
        return getattr(self.app, "last_demodulation", None)

    def _load_a(self) -> None:
        audio = self._ask_wav("Cargar el audio original")
        if audio is None:
            return
        self._a_file = audio
        self._a_file_label = f"Archivo: {audio.name}"
        self.a_combo.configure(values=self._values_a())
        self.a_var.set(self._a_file_label)
        self._on_source_changed()

    def _load_b(self) -> None:
        audio = self._ask_wav("Cargar el audio recuperado")
        if audio is None:
            return
        self._b_file = audio
        self._b_file_label = f"Archivo: {audio.name}"
        self.b_combo.configure(values=self._values_b())
        self.b_var.set(self._b_file_label)
        self._on_source_changed()

    def _ask_wav(self, title: str) -> Audio | None:
        path = filedialog.askopenfilename(
            parent=self,
            title=title,
            initialdir=str(output_dir()),
            filetypes=[("Audio WAV", "*.wav"), ("Todos los archivos", "*.*")],
        )
        if not path:
            return None
        try:
            return read_wav(path)
        except AudioError as exc:
            messagebox.showerror(title, str(exc))
            self.app.set_status(str(exc), error=True)
            return None

    def _use_latest(self) -> None:
        self._a_file = None
        self._b_file = None
        self._a_file_label = ""
        self._b_file_label = ""
        self.a_combo.configure(values=self._values_a())
        self.b_combo.configure(values=self._values_b())
        self.a_var.set(A_TX)
        self.b_var.set(B_DEMOD)
        self._invalidate_result()
        self._update_sources()
        self.app.set_status("Se usarán la última transmisión y la última "
                            "demodulación disponibles.")

    def _on_source_changed(self, _event=None) -> None:
        self._invalidate_result()
        self._update_sources()

    def _on_view_changed(self, _event=None) -> None:
        if self._result is not None:
            self.refresh()

    # ------------------------------------------------------------------
    # Ciclo de vida
    # ------------------------------------------------------------------
    def on_show(self) -> None:
        self._update_sources()
        super().on_show()

    def _update_sources(self) -> None:
        a = self._resolve_a()
        b = self._resolve_b()

        if self._result is not None and (a is not self._a_audio
                                         or b is not self._b_audio):
            self._invalidate_result()

        self.sources_var.set(f"A: {self._describe(a)}    |    "
                             f"B: {self._describe(b)}")
        state = "normal" if a is not None and b is not None else "disabled"
        self.compare_btn.configure(state=state)
        if self._result is None:
            for btn in (self.play_a_btn, self.play_b_btn, self.play_ab_btn,
                        self.csv_btn, self.save_ab_btn):
                btn.configure(state="disabled")

    @staticmethod
    def _describe(audio: Audio | None) -> str:
        if audio is None:
            return "(sin audio)"
        return (f"{audio.name} · {audio.duration:.3f} s · {audio.fs} Hz · "
                f"{len(audio.x)} muestras")

    # ------------------------------------------------------------------
    # Comparación
    # ------------------------------------------------------------------
    def _compare(self) -> None:
        a = self._resolve_a()
        b = self._resolve_b()
        if a is None or b is None:
            messagebox.showinfo(
                "Comparar",
                "Seleccione el audio original y el audio recuperado, o cargue "
                "un WAV para cada uno.")
            return
        try:
            delay = float(self.delay_var.get().replace(",", "."))
            if delay < 0.0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Comparar",
                                 "La búsqueda de desfase debe ser un número ≥ 0.")
            return

        self._stop_play()
        try:
            result = compare_audio(
                a.x, a.fs, b.x, b.fs,
                align=bool(self.align_var.get()),
                match_gain=bool(self.gain_var.get()),
                max_delay_s=delay,
            )
        except ValueError as exc:
            messagebox.showerror("Comparar", str(exc))
            self.app.set_status(str(exc), error=True)
            return
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Comparar",
                                 f"No se pudo comparar los audios:\n{exc}")
            self.app.set_status(f"Error al comparar: {exc}", error=True)
            return

        if result.duration < 0.010:
            messagebox.showwarning(
                "Comparar",
                "El tramo común es demasiado corto. Revise el desfase o los "
                "audios seleccionados.")
            return

        self._a_audio = a
        self._b_audio = b
        self._result = result
        self.info_var.set(self._format_info(a, b, result))
        for btn in (self.play_a_btn, self.play_b_btn, self.play_ab_btn,
                    self.csv_btn, self.save_ab_btn):
            btn.configure(state="normal")
        self.app.set_status(
            f"Comparación lista: correlación {result.corr:.4f}, "
            f"SNR {self._fmt_snr(result.snr_db)}.")
        self.mark_dirty()
        self.refresh()

    def _invalidate_result(self) -> None:
        self._stop_play()
        self._result = None
        self._a_audio = None
        self._b_audio = None
        self.info_var.set("")
        for btn in (self.play_a_btn, self.play_b_btn, self.play_ab_btn,
                    self.csv_btn, self.save_ab_btn):
            btn.configure(state="disabled")

    def _format_info(self, a: Audio, b: Audio,
                     r: ComparisonResult) -> str:
        resampled = ""
        if r.resampled:
            resampled = (f" · B remuestreada de {r.fs_test_original:.0f} Hz "
                         f"a {r.fs:.0f} Hz")
        if r.delay_samples == 0:
            delay = "sin desfase"
        elif r.delay_samples > 0:
            delay = (f"B llega {r.delay_samples} muestras después "
                     f"(+{1e3 * r.delay_s:.2f} ms)")
        else:
            delay = (f"B llega {-r.delay_samples} muestras antes "
                     f"({1e3 * r.delay_s:.2f} ms)")
        # Los nombres, duraciones y fs de A y B ya están en la línea de
        # orígenes; aquí solo van las métricas de la comparación para no
        # robarle altura a las gráficas.
        lines = [
            f"Tramo: {r.duration:.3f} s ({100.0 * r.overlap_ref:.1f} % de A, "
            f"{100.0 * r.overlap_test:.1f} % de B){resampled} · {delay} · "
            f"ganancia aplicada a B: {r.gain:.4f}",
            f"Correlación: {r.corr:.5f} ({r.correlation_percent:.2f} %) · "
            f"fidelidad {r.fidelity} · SNR: {self._fmt_snr(r.snr_db)} · "
            f"RMSE: {r.rmse:.5f} · error máx.: {r.max_error:.5f} · "
            f"RMS A/B: {r.rms_ref:.5f} / {r.rms_test:.5f}",
        ]
        return "\n".join(lines)

    @staticmethod
    def _fmt_snr(value: float) -> str:
        if math.isinf(value):
            return "∞" if value > 0 else "−∞"
        if math.isnan(value):
            return "—"
        return f"{value:.2f} dB"

    # ------------------------------------------------------------------
    # Reproducción
    # ------------------------------------------------------------------
    def _play(self, which: str) -> None:
        a, b = self._a_audio, self._b_audio
        if a is None or b is None:
            return
        self._stop_play()
        try:
            if which == "a":
                player = Player(a.x, a.fs)
                status = f"Reproduciendo A (original) a {a.fs} Hz…"
            elif which == "b":
                player = Player(b.x, b.fs)
                status = f"Reproduciendo B (recuperado) a {b.fs} Hz…"
            else:
                seq, fs, start_b = sequential_signal(a.x, a.fs, b.x, b.fs)
                player = Player(seq, fs)
                status = (f"Reproduciendo A y luego B, sin superposición "
                          f"(B empieza en {start_b:.2f} s)…")
        except ValueError as exc:
            messagebox.showerror("Reproducir", str(exc))
            return
        self._player = player
        player.start()
        for btn in (self.play_a_btn, self.play_b_btn, self.play_ab_btn):
            btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.app.set_status(status)
        self._poll_player()

    def _poll_player(self) -> None:
        self._poll_id = None
        p = self._player
        if p is None:
            return
        if p.error:
            self.app.set_status(p.error, error=True)
            self._player = None
            self._restore_play_buttons()
            return
        if p.is_playing:
            try:
                self._poll_id = self.after(150, self._poll_player)
            except tk.TclError:
                self._poll_id = None
            return
        self._player = None
        self._restore_play_buttons()
        self.app.set_status("Reproducción terminada.")

    def _stop_play(self) -> None:
        if self._poll_id is not None:
            try:
                self.after_cancel(self._poll_id)
            except tk.TclError:
                pass
            self._poll_id = None
        if self._player is not None:
            self._player.stop()
            self._player = None
        self._restore_play_buttons()

    def _restore_play_buttons(self) -> None:
        self.stop_btn.configure(state="disabled")
        if self._result is None:
            return
        self.play_a_btn.configure(state="normal")
        self.play_b_btn.configure(state="normal")
        self.play_ab_btn.configure(state="normal")

    # ------------------------------------------------------------------
    # Exportación
    # ------------------------------------------------------------------
    def _save_csv(self) -> None:
        r = self._result
        if r is None:
            return
        step = max(1, int(math.ceil(r.n / MAX_CSV_ROWS)))
        rows = [
            (f"{r.t[i]:.6f}", f"{r.reference[i]:.8f}", f"{r.test[i]:.8f}",
             f"{r.error[i]:.8f}")
            for i in range(0, r.n, step)
        ]
        path = save_rows_csv(
            self, "comparacion_original_recuperado",
            ["tiempo_s", "A_original", "B_recuperado", "error_B_menos_A"],
            rows)
        if path:
            self.app.set_status(f"Comparación exportada en {path}")

    def _save_sequential(self) -> None:
        a, b = self._a_audio, self._b_audio
        if a is None or b is None:
            return
        try:
            seq, fs, start_b = sequential_signal(a.x, a.fs, b.x, b.fs)
        except ValueError as exc:
            messagebox.showerror("Guardar A → B", str(exc))
            return
        path = filedialog.asksaveasfilename(
            parent=self,
            title="Guardar la comparación secuencial A → B",
            initialdir=str(output_dir()),
            initialfile=timestamp("comparacion_secuencial", "wav"),
            defaultextension=".wav",
            filetypes=[("Audio WAV", "*.wav"), ("Todos los archivos", "*.*")],
        )
        if not path:
            return
        try:
            save_wav(path, seq, fs, bits=16)
        except (AudioError, ValueError) as exc:
            messagebox.showerror("Guardar A → B", f"No se pudo guardar:\n{exc}")
            self.app.set_status(f"Error al guardar A → B: {exc}", error=True)
            return
        self.app.set_status(
            f"Comparación secuencial guardada en {path} "
            f"(B empieza en {start_b:.2f} s).")

    # ------------------------------------------------------------------
    # Gráficas
    # ------------------------------------------------------------------
    def refresh(self) -> None:
        r = self._result
        if r is None:
            a = self._resolve_a()
            b = self._resolve_b()
            if a is not None and b is not None:
                text = ("Pulse «Comparar» para ver el original y el recuperado "
                        "en ejes separados.")
            else:
                text = ("Falta el audio original o el recuperado.\n"
                        "Transmita y demodule un audio, o cargue un WAV en A "
                        "y otro en B.")
            self.message(text)
            return
        if self.view_var.get() == "Espectro":
            self._plot_spectrum(r)
        else:
            self._plot_time(r)

    def _plot_time(self, r: ComparisonResult) -> None:
        ax1, ax2, ax3 = self.axes(3, height_ratios=[1.0, 1.0, 0.75])
        t = r.t

        ta, xa = display_slice(t, r.reference, 12000)
        ax1.plot(ta, xa, color="#1f77b4", lw=0.7, label="A · original")
        ax1.set_ylabel("Amplitud")
        ax1.set_title(f"A · audio original · fs = {r.fs:.0f} Hz · "
                      f"{r.duration:.3f} s")
        ax1.grid(alpha=0.25)
        ax1.legend(loc="upper right", fontsize=8)

        tb, xb = display_slice(t, r.test, 12000)
        ax2.plot(tb, xb, color="#ff7f0e", lw=0.7,
                 label="B · recuperado (alineado)")
        ax2.set_ylabel("Amplitud")
        ax2.set_title(f"B · audio recuperado · fs = {r.fs:.0f} Hz · "
                      f"{r.duration:.3f} s")
        ax2.grid(alpha=0.25)
        ax2.legend(loc="upper right", fontsize=8)

        te, xe = display_slice(t, r.error, 12000)
        ax3.plot(te, xe, color="#d62728", lw=0.6, label="B − A")
        ax3.axhline(0.0, color="#888888", lw=0.5)
        ax3.axhline(r.rmse, color="#555555", ls="--", lw=0.7,
                    label=f"±RMSE = {r.rmse:.4f}")
        ax3.axhline(-r.rmse, color="#555555", ls="--", lw=0.7)
        ax3.set_xlabel("Tiempo [s]")
        ax3.set_ylabel("Error")
        ax3.set_title(f"Diferencia B − A · SNR = {self._fmt_snr(r.snr_db)} · "
                      f"correlación = {r.corr:.4f}")
        ax3.grid(alpha=0.25)
        ax3.legend(loc="upper right", fontsize=8)

        xmax = max(r.duration, 1e-3)
        for ax in (ax1, ax2, ax3):
            ax.set_xlim(0.0, xmax)
        self.draw()

    def _plot_spectrum(self, r: ComparisonResult) -> None:
        n = min(r.n, MAX_FFT)
        window = self.app.window_var.get()
        f, ma = spectrum(r.reference[:n], r.fs, window)
        _, mb = spectrum(r.test[:n], r.fs, window)
        if f.size < 8:
            self.message("El tramo comparado es demasiado corto para el "
                         "espectro.")
            return
        da, dbs = db(ma), db(mb)

        ax1, ax2, ax3 = self.axes(3, height_ratios=[1.0, 1.0, 0.75])
        ax1.plot(f, da, color="#1f77b4", lw=0.8, label="A · original")
        ax1.set_ylabel("Magnitud [dB]")
        ax1.set_title(f"Espectro de A · ventana {window} · "
                      f"{n} muestras")
        ax1.grid(alpha=0.25)
        ax1.legend(loc="upper right", fontsize=8)

        ax2.plot(f, dbs, color="#ff7f0e", lw=0.8, label="B · recuperado")
        ax2.set_ylabel("Magnitud [dB]")
        ax2.set_title("Espectro de B · recuperado")
        ax2.grid(alpha=0.25)
        ax2.legend(loc="upper right", fontsize=8)

        diff = dbs - da
        ax3.plot(f, diff, color="#d62728", lw=0.7, label="B − A")
        ax3.axhline(0.0, color="#888888", lw=0.5)
        ax3.set_xlabel("Frecuencia [Hz]")
        ax3.set_ylabel("Diferencia [dB]")
        ax3.set_title("Diferencia de magnitud B − A")
        ax3.grid(alpha=0.25)
        ax3.legend(loc="upper right", fontsize=8)

        xmax = min(8000.0, 0.5 * r.fs)
        for ax in (ax1, ax2, ax3):
            ax.set_xlim(0.0, xmax)
        self.draw()

    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        self._stop_play()
