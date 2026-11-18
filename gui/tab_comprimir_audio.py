"""Pestaña 9 (Desafío 4, punto 1): compresión de audio WAV ↔ FLAC.

Comprime el audio grabado en el Desafío 2 a FLAC (sin pérdida), permite
descomprimirlo a WAV y verificar que las muestras son idénticas bit a bit.
"""
from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np

from core.audio_io import AudioError, Player, read_wav, save_wav
from core.compression_audio import (AudioCompressionResult, CompressionError,
                                    compress_audio_wav, decompress_audio_flac,
                                    verify_audio_roundtrip)
from core.dsp import display_slice

from .widgets import PlotTab, output_dir, timestamp

SRC_CURRENT = "Audio actual de RETICO"
SRC_FILE = "Archivo WAV..."


class TabComprimirAudio(PlotTab):
    """Compresión y reconstrucción del audio del Desafío 2."""

    def __init__(self, master, app):
        super().__init__(master, app)
        self._src_manual: Path | None = None
        self._src_wav: Path | None = None
        self._flac: Path | None = None
        self._restored: Path | None = None
        self._result: AudioCompressionResult | None = None
        self._player: Player | None = None
        self._poll_id: str | None = None

        self.source_var = tk.StringVar(value=SRC_CURRENT)
        self.info_var = tk.StringVar(
            value="Comprima el audio del Desafío 2 a FLAC y verifique la reconstrucción.")
        self._build_controls()
        self._update_buttons()
        self.message("Cargue o grabe audio en la pestaña 1, o elija un WAV,\n"
                     "y pulse «Comprimir a FLAC».")

    # ------------------------------------------------------------------
    def _build_controls(self) -> None:
        c = self.controls
        row1 = ttk.Frame(c)
        row1.pack(fill="x")
        ttk.Label(row1, text="Origen").pack(side="left", padx=(0, 4))
        self.source_combo = ttk.Combobox(
            row1, textvariable=self.source_var, state="readonly", width=25,
            values=[SRC_CURRENT, SRC_FILE])
        self.source_combo.pack(side="left", padx=(0, 4))
        self.source_combo.bind("<<ComboboxSelected>>", self._on_source_changed)
        self.pick_btn = ttk.Button(row1, text="📂 Elegir WAV", width=13,
                                   command=self._pick_wav)
        self.pick_btn.pack(side="left", padx=(0, 12))
        self.compress_btn = ttk.Button(row1, text="🗜 Comprimir a FLAC",
                                       command=self._compress)
        self.compress_btn.pack(side="left", padx=(0, 8))
        self.decompress_btn = ttk.Button(row1, text="📂 Descomprimir FLAC",
                                         command=self._decompress,
                                         state="disabled")
        self.decompress_btn.pack(side="left", padx=(0, 8))
        self.verify_btn = ttk.Button(row1, text="✅ Verificar", width=11,
                                     command=self._verify, state="disabled")
        self.verify_btn.pack(side="left", padx=(0, 8))

        row2 = ttk.Frame(c)
        row2.pack(fill="x", pady=(6, 0))
        self.play_a_btn = ttk.Button(row2, text="▶ Original", width=11,
                                     command=lambda: self._play("original"),
                                     state="disabled")
        self.play_a_btn.pack(side="left", padx=(0, 4))
        self.play_b_btn = ttk.Button(row2, text="▶ Reconstruido", width=14,
                                     command=lambda: self._play("restored"),
                                     state="disabled")
        self.play_b_btn.pack(side="left", padx=(0, 4))
        self.stop_btn = ttk.Button(row2, text="■ Detener", width=9,
                                   command=self._stop_play, state="disabled")
        self.stop_btn.pack(side="left", padx=(0, 12))
        ttk.Button(row2, text="🔎 Abrir salidas", width=15,
                   command=self._open_outputs).pack(side="left")

        ttk.Label(c, textvariable=self.info_var, style="Info.TLabel",
                  wraplength=1240, justify="left").pack(anchor="w", pady=(6, 0))
        ttk.Label(
            c,
            text="FLAC guarda las mismas muestras PCM que el WAV: la reconstrucción "
                 "es exacta (bit a bit) y el hash SHA-256 de las muestras coincide. "
                 "El FLAC se guarda en salidas/; cualquier compañero que descargue "
                 "el repositorio puede abrirlo con «Descomprimir FLAC» y verificarlo.",
            style="Subtitle.TLabel", wraplength=1240, justify="left").pack(
            anchor="w", pady=(4, 0))

    # ------------------------------------------------------------------
    def _on_source_changed(self, _event=None) -> None:
        self.mark_dirty()
        if self.source_var.get() == SRC_FILE and self._src_manual is None:
            self._pick_wav()

    def _pick_wav(self) -> None:
        path = filedialog.askopenfilename(
            parent=self, title="Abrir WAV del Desafío 2",
            initialdir=str(output_dir()),
            filetypes=[("Audio WAV", "*.wav *.WAV"), ("Todos los archivos", "*.*")])
        if not path:
            if self._src_manual is None:
                self.source_var.set(SRC_CURRENT)
            return
        self._src_manual = Path(path)
        self.source_var.set(SRC_FILE)
        self._src_wav = self._src_manual
        self._flac = None
        self._restored = None
        self._result = None
        self.info_var.set(f"WAV de origen: {self._src_manual.name}")
        self.app.set_status(f"WAV seleccionado: {path}")
        self.mark_dirty()
        self.on_show()
        self._update_buttons()

    def _source_wav(self) -> Path | None:
        if self.source_var.get() == SRC_FILE:
            if self._src_manual is None or not self._src_manual.is_file():
                messagebox.showinfo("Comprimir", "Elija primero un archivo WAV.",
                                    parent=self)
                return None
            return self._src_manual
        audio = self.app.audio
        if audio is None:
            messagebox.showinfo("Comprimir", "Grabe o cargue audio en la pestaña 1.",
                                parent=self)
            return None
        if audio.path and Path(audio.path).is_file() \
                and Path(audio.path).suffix.lower() == ".wav":
            return Path(audio.path)
        # El audio viene de una demodulación o no tiene archivo: exportamos una
        # copia PCM de 16 bits para poder comprimirlo con soundfile.
        dest = output_dir() / timestamp("audio_para_comprimir", "wav")
        try:
            save_wav(str(dest), audio.x, audio.fs, bits=16)
        except AudioError as exc:
            messagebox.showerror("Comprimir", str(exc), parent=self)
            return None
        self.app.set_status(f"Audio exportado a {dest.name} para comprimirlo.")
        return dest

    # ------------------------------------------------------------------
    def _compress(self) -> None:
        src = self._source_wav()
        if src is None:
            return
        dst = filedialog.asksaveasfilename(
            parent=self, title="Guardar audio comprimido (FLAC)",
            initialdir=str(output_dir()),
            initialfile=timestamp("audio_comprimido", "flac"),
            defaultextension=".flac",
            filetypes=[("Audio FLAC", "*.flac"), ("Todos los archivos", "*.*")])
        if not dst:
            return
        try:
            result = compress_audio_wav(src, dst)
        except CompressionError as exc:
            messagebox.showerror("Comprimir audio", str(exc), parent=self)
            self.app.set_status(str(exc), error=True)
            return
        self._src_wav, self._flac, self._restored = src, Path(dst), None
        self._result = result
        self.info_var.set(result.summary())
        self.app.set_status(
            f"Audio comprimido: {result.original_size:,} → {result.compressed_size:,} "
            f"bytes ({result.ratio:.2f}×).")
        self.mark_dirty()
        self.on_show()
        self._update_buttons()

    def _decompress(self) -> None:
        if self._flac is None:
            return
        src = filedialog.askopenfilename(
            parent=self, title="Abrir FLAC a descomprimir",
            initialdir=str(self._flac.parent), initialfile=self._flac.name,
            filetypes=[("Audio FLAC", "*.flac"), ("Todos los archivos", "*.*")])
        if not src:
            return
        dst = filedialog.asksaveasfilename(
            parent=self, title="Guardar audio reconstruido (WAV)",
            initialdir=str(output_dir()),
            initialfile=timestamp("audio_reconstruido", "wav"),
            defaultextension=".wav", filetypes=[("WAV PCM", "*.wav")])
        if not dst:
            return
        same = bool(self._result and Path(src) == self._flac)
        expected = self._result.pcm_sha256 if same else ""
        subtype = self._result.flac_subtype if same else ""
        try:
            result = decompress_audio_flac(
                src, dst, expected_sha256=expected, restored_subtype=subtype)
        except CompressionError as exc:
            messagebox.showerror("Descomprimir audio", str(exc), parent=self)
            self.app.set_status(str(exc), error=True)
            return
        if same and self._result is not None:
            result.original_size = self._result.original_size
            result.compressed = self._flac
            result.subtype = self._result.subtype
            result.frames = self._result.frames
        self._restored = Path(dst)
        self._result = result
        self.info_var.set(result.summary())
        self.app.set_status(
            f"Audio reconstruido en {Path(dst).name}. "
            f"{'Muestras EXACTAS.' if result.exact else 'Sin hash de referencia o con diferencias.'}")
        self.mark_dirty()
        self.on_show()
        self._update_buttons()

    def _verify(self) -> None:
        if self._src_wav is None or self._restored is None:
            return
        original = filedialog.askopenfilename(
            parent=self, title="WAV original para verificar",
            initialdir=str(self._src_wav.parent), initialfile=self._src_wav.name,
            filetypes=[("Audio WAV", "*.wav")])
        if not original:
            return
        restored = filedialog.askopenfilename(
            parent=self, title="WAV reconstruido para verificar",
            initialdir=str(self._restored.parent), initialfile=self._restored.name,
            filetypes=[("Audio WAV", "*.wav")])
        if not restored:
            return
        try:
            checked = verify_audio_roundtrip(original, restored)
        except CompressionError as exc:
            messagebox.showerror("Verificar audio", str(exc), parent=self)
            return
        if self._result is not None:
            checked.original_size = self._result.original_size
            checked.compressed = self._flac
            checked.compressed_size = self._result.compressed_size
        self._result = checked
        self.info_var.set(checked.summary())
        self.app.set_status("Verificación: " + ("muestras idénticas (bit a bit)."
                             if checked.exact else "hay diferencias."))
        self.mark_dirty()
        self.on_show()

    # ------------------------------------------------------------------
    def _play(self, which: str) -> None:
        path = self._src_wav if which == "original" else self._restored
        if path is None or not path.is_file():
            return
        self._stop_play()
        try:
            audio = read_wav(str(path))
        except AudioError as exc:
            messagebox.showerror("Reproducir", str(exc), parent=self)
            return
        self._player = Player(audio.x, audio.fs)
        self._player.start()
        self.play_a_btn.configure(state="disabled")
        self.play_b_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.app.set_status(f"Reproduciendo {path.name}…")
        self._poll_player()

    def _poll_player(self) -> None:
        self._poll_id = None
        player = self._player
        if player is None:
            return
        if player.error:
            self.app.set_status(player.error, error=True)
            self._stop_play()
            return
        if player.is_playing:
            try:
                self._poll_id = self.after(150, self._poll_player)
            except tk.TclError:
                self._poll_id = None
            return
        self._stop_play()
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
        self._update_buttons()

    def _update_buttons(self) -> None:
        has_flac = self._flac is not None and self._flac.is_file()
        has_restored = self._restored is not None and self._restored.is_file()
        self.decompress_btn.configure(state="normal" if has_flac else "disabled")
        self.verify_btn.configure(state="normal" if has_restored else "disabled")
        playable = self._src_wav is not None and self._src_wav.is_file()
        self.play_a_btn.configure(state="normal" if playable else "disabled")
        self.play_b_btn.configure(state="normal" if has_restored else "disabled")
        self.stop_btn.configure(state="normal" if self._player is not None else "disabled")

    def _open_outputs(self) -> None:
        import os
        import subprocess
        folder = output_dir()
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(folder))  # noqa: S606
            else:  # pragma: no cover
                subprocess.Popen(["xdg-open", str(folder)])
        except Exception as exc:  # noqa: BLE001
            self.app.set_status(f"No se pudo abrir {folder}: {exc}", error=True)

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        src = self._src_wav
        if src is None or not src.is_file():
            if self.app.audio is None:
                self.message("Todavía no hay audio.\n"
                             "Use la pestaña 1 o elija un WAV con «Elegir WAV».")
            else:
                self.message("Pulse «Comprimir a FLAC» para comprimir el audio actual.")
            return
        try:
            original = read_wav(str(src))
        except AudioError as exc:
            self.message(f"No se pudo leer el WAV de origen:\n{exc}")
            return
        restored = None
        if self._restored is not None and self._restored.is_file():
            try:
                restored = read_wav(str(self._restored))
            except AudioError:
                restored = None

        axes = self.axes(2 if restored is None else 3)
        t, x = display_slice(np.arange(len(original.x)) / original.fs,
                             original.x, 12000)
        axes[0].plot(t, x, lw=0.6, color="#1f77b4")
        axes[0].set_title(f"Original · {src.name}")
        if restored is not None:
            tr, xr = display_slice(np.arange(len(restored.x)) / restored.fs,
                                   restored.x, 12000)
            axes[1].plot(tr, xr, lw=0.6, color="#2ca02c")
            axes[1].set_title(f"Reconstruido desde FLAC · {self._restored.name}")
            n = min(len(original.x), len(restored.x))
            diff = np.abs(original.x[:n] - restored.x[:n])
            axes[2].plot(np.arange(n) / original.fs, diff, lw=0.6, color="#d62728")
            axes[2].set_title(f"|error| · máx. {diff.max():.6g}" if n else "|error|")
            axes[2].set_ylim(bottom=0.0)
        else:
            axes[1].axis("off")
            axes[1].set_title("Descomprima el FLAC para ver la comparación")
        for ax in np.atleast_1d(axes):
            ax.grid(alpha=0.25)
            ax.set_xlabel("Tiempo [s]")
            ax.set_ylabel("Amplitud")
        self.draw()

    def shutdown(self) -> None:
        self._stop_play()
