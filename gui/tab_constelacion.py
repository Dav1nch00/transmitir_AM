"""Pestaña 10: constelaciones y ejemplo de una palabra PCM de 5 bits."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np

from core.constellation import (ALLOWED_BITS, TECHNIQUES,
                                ConstellationError, constellation, pcm_word)

from .widgets import PlotTab, save_rows_csv

_DESCRIPTIONS = {
    "PCM (PAM)": (
        "PAM de M niveles centrados: cada palabra PCM se transmite como una "
        "amplitud distinta sobre el eje I (sin portadora o con ella)."),
    "ASK": (
        "Amplitud unipolar 0 … 1 sobre el eje I: el símbolo 0 se apaga y el "
        "último usa la amplitud máxima."),
    "PSK": (
        "Fase 2πk/M sobre la circunferencia unitaria; todos los símbolos "
        "tienen la misma energía."),
    "QAM": (
        "Rejilla cuadrada I/Q: cada símbolo combina una amplitud en fase y "
        "otra en cuadratura."),
    "FSK": (
        "M tonos equiespaciados: el símbolo elige una frecuencia instantánea "
        "alrededor de la portadora."),
    "MFSK": (
        "Variante M-aria de FSK: M tonos equiespaciados, con separación "
        "Δf menor cuanto mayor es M."),
    "GFSK": (
        "FSK con filtro gaussiano (BT): los tonos nominales son los mismos "
        "que en FSK, pero las transiciones son suaves."),
}


class TabConstelacion(PlotTab):
    """Constelación de cada técnica y ejemplo palabra PCM → punto PAM."""

    def __init__(self, master, app):
        super().__init__(master, app)
        self.technique_var = tk.StringVar(value="PCM (PAM)")
        self.bits_var = tk.StringVar(value="5")
        self.word_var = tk.StringVar(value="10110")
        self.labels_var = tk.BooleanVar(value=True)
        self.grid_var = tk.BooleanVar(value=True)
        self.info_var = tk.StringVar(value="Sin datos.")
        self._build_controls()
        self._build_table()
        self._update_word_state()
        self.refresh()

    # ------------------------------------------------------------------
    def _build_controls(self) -> None:
        c = self.controls
        ttk.Label(c, text="Técnica").grid(row=0, column=0, sticky="w")
        self.tech_combo = ttk.Combobox(c, textvariable=self.technique_var,
                                       width=14, state="readonly",
                                       values=list(TECHNIQUES))
        self.tech_combo.grid(row=0, column=1, padx=(4, 12))
        self.tech_combo.bind("<<ComboboxSelected>>", self._on_technique_changed)

        ttk.Label(c, text="Bits/símbolo").grid(row=0, column=2, sticky="w")
        self.bits_combo = ttk.Combobox(c, textvariable=self.bits_var,
                                       width=5, state="readonly",
                                       values=self._bits_values())
        self.bits_combo.grid(row=0, column=3, padx=(4, 12))
        self.bits_combo.bind("<<ComboboxSelected>>", self._on_bits_changed)

        ttk.Label(c, text="Palabra PCM").grid(row=0, column=4, sticky="w")
        self.word_entry = ttk.Entry(c, textvariable=self.word_var, width=10)
        self.word_entry.grid(row=0, column=5, padx=(4, 4))
        self.word_entry.bind("<Return>", self._on_word_applied)
        self.word_entry.bind("<FocusOut>", self._on_word_applied)
        self.word_button = ttk.Button(c, text="Aplicar",
                                      command=self._on_word_applied)
        self.word_button.grid(row=0, column=6, padx=(0, 12))

        ttk.Checkbutton(c, text="Etiquetas binarias",
                        variable=self.labels_var,
                        command=self._on_control_changed).grid(
            row=0, column=7, padx=(0, 10), sticky="w")
        ttk.Checkbutton(c, text="Rejilla", variable=self.grid_var,
                        command=self._on_control_changed).grid(
            row=0, column=8, padx=(0, 12), sticky="w")
        ttk.Button(c, text="Actualizar", command=self.refresh).grid(
            row=0, column=9, sticky="w")

        ttk.Label(c, textvariable=self.info_var, style="Info.TLabel",
                  wraplength=1240, justify="left").grid(
            row=1, column=0, columnspan=10, sticky="w", pady=(6, 2))

    def _build_table(self) -> None:
        self.table_frame = ttk.Frame(self.footer, padding=(10, 2, 10, 4))
        self.table_frame.pack(fill="x")
        left = ttk.Frame(self.table_frame)
        left.pack(side="left", fill="both", expand=True)

        self.columns = ("k", "bits", "c1", "c2", "c3", "c4")
        self.table = ttk.Treeview(left, columns=self.columns, show="headings",
                                  height=4)
        for col in self.columns:
            self.table.heading(col, text=col)
            self.table.column(col, width=95, anchor="center", stretch=False)
        self.table.tag_configure("selected", background="#ffe08a")
        scroll = ttk.Scrollbar(left, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=scroll.set)
        self.table.pack(side="left", fill="both", expand=True)
        scroll.pack(side="left", fill="y")

        right = ttk.Frame(self.table_frame)
        right.pack(side="right", fill="y", padx=(8, 0))
        ttk.Button(right, text="Exportar CSV",
                   command=self._export_csv).pack(fill="x", pady=1)
        ttk.Label(right, text="Tabla de símbolos de la\nconstelación mostrada.",
                  style="Subtitle.TLabel", justify="left").pack(anchor="w", pady=(6, 0))

    # ------------------------------------------------------------------
    def _bits_values(self) -> list[str]:
        technique = self.technique_var.get()
        return [str(b) for b in ALLOWED_BITS.get(technique, (1, 2, 3, 4))]

    def _update_word_state(self) -> None:
        state = "normal" if self.technique_var.get() == "PCM (PAM)" else "disabled"
        self.word_entry.configure(state=state)
        self.word_button.configure(state=state)

    def _sync_word_length(self) -> None:
        """Ajusta la palabra al número de bits elegido."""
        try:
            bits = int(self.bits_var.get())
        except (tk.TclError, ValueError):
            return
        text = self.word_var.get().strip()
        if len(text) == bits and text and all(ch in "01" for ch in text):
            return
        try:
            value = int(text, 2) if text else 0b10110
        except ValueError:
            value = 0b10110
        value &= (1 << bits) - 1
        self.word_var.set(format(value, f"0{bits}b"))

    def _on_technique_changed(self, _event=None) -> None:
        technique = self.technique_var.get()
        if technique == "PCM (PAM)":
            self.bits_var.set("5")
        values = self._bits_values()
        if self.bits_var.get() not in values:
            self.bits_var.set("4" if technique == "QAM" else "2")
        self.bits_combo.configure(values=values)
        self._sync_word_length()
        self._update_word_state()
        self.refresh()
        self.app.set_status("Constelación actualizada.")

    def _on_bits_changed(self, _event=None) -> None:
        self._sync_word_length()
        self.refresh()
        self.app.set_status("Constelación actualizada.")

    def _on_word_applied(self, _event=None) -> None:
        self.refresh()
        self.app.set_status("Palabra PCM aplicada a la constelación.")

    def _on_control_changed(self, _event=None) -> None:
        self.refresh()
        self.app.set_status("Constelación actualizada.")

    # ------------------------------------------------------------------
    def _vref(self) -> float:
        try:
            vref = float(self.app.vref_var.get())
        except (tk.TclError, ValueError):
            return 1.0
        return vref if vref > 0.0 else 1.0

    def _selected_word(self, c):
        """Palabra PCM válida para la constelación actual, o ``None``."""
        if c.technique != "PCM (PAM)":
            return None, None
        try:
            return pcm_word(self.word_var.get(), c.bits_per_symbol,
                            vref=self._vref()), None
        except ConstellationError as exc:
            return None, str(exc)

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        try:
            c = constellation(self.technique_var.get(), int(self.bits_var.get()))
        except (ConstellationError, ValueError) as exc:
            self.info_var.set(f"No se pudo construir la constelación: {exc}")
            self.message(str(exc))
            self._clear_table()
            return

        word, word_error = self._selected_word(c)
        self._draw_constellation(c, word)
        self._fill_table(c, word)
        self._set_info(c, word, word_error)

    def _set_info(self, c, word, word_error) -> None:
        desc = _DESCRIPTIONS.get(c.technique, "")
        if c.is_iq:
            metrics = (
                f"E_media = {c.mean_energy:.4f} · E_pico = {c.peak_power:.4f} · "
                f"PAPR = {c.papr_db:.2f} dB · d_min = {c.min_distance:.4f} · "
                f"d_min/√E = {c.dmin_normalized:.4f}")
        else:
            spacing = (2.0 / (c.m - 1)) if c.m > 1 else 0.0
            metrics = (f"separación normalizada Δf = {spacing:.4f} · "
                       f"banda nominal = ±1")

        if c.technique == "PCM (PAM)":
            header = (f"PCM (PAM) · M = 2^{c.bits_per_symbol} = {c.m} niveles · "
                      f"{c.bits_per_symbol} bit(s)/muestra · {metrics}")
            if word is not None:
                self.info_var.set(
                    f"{header}\n"
                    f"Palabra {word.word} (MSB primero) → código PCM k = "
                    f"{word.code} → nivel PAM = {word.level:+.4f} · "
                    f"q = k − M/2 = {word.signed_code:+d} → tensión = "
                    f"q·2·Vref/M = {word.volts:+.4f} V "
                    f"(Vref = {word.vref:g} V)\n"
                    f"La constelación reparte M niveles simétricos; el ADC usa "
                    f"q = k − M/2 (mid-tread), por eso el nivel del punto y la "
                    f"tensión difieren en medio LSB. Abajo se muestran los "
                    f"{word.bits} bits de la palabra en serie (MSB primero).\n"
                    f"{desc}")
            else:
                self.info_var.set(
                    f"{header}\n"
                    f"Palabra «{self.word_var.get()}» no válida: {word_error}\n"
                    f"Escriba exactamente {c.bits_per_symbol} bits (0/1) para "
                    f"resaltar su punto en la constelación.\n{desc}")
            return

        self.info_var.set(
            f"{c.technique} · M = 2^{c.bits_per_symbol} = {c.m} símbolos · "
            f"{c.bits_per_symbol} bit(s)/símbolo\n{metrics}\n{desc}")

    # ------------------------------------------------------------------
    def _draw_constellation(self, c, word) -> None:
        if c.technique == "PCM (PAM)":
            ax1, ax2 = self.axes(2, height_ratios=[1.9, 0.9])
            self._draw_pcm(ax1, c, word)
            self._draw_word_bits(ax2, c, word)
        else:
            ax = self.axes(1)
            if self.grid_var.get():
                ax.grid(alpha=0.25)
            if c.is_iq:
                self._draw_iq(ax, c)
            else:
                self._draw_fsk(ax, c)
        self.draw()

    def _draw_pcm(self, ax, c, word) -> None:
        if self.grid_var.get():
            ax.grid(alpha=0.25)
        k = np.arange(c.m)
        level = c.i_values
        ax.axhline(0.0, color="#999999", lw=0.7)
        ax.vlines(k, 0.0, level, color="#c7c7c7", lw=0.8)
        ax.scatter(k, level, s=30, color="#1f77b4", edgecolor="#0b3d61",
                   zorder=3)
        if self.labels_var.get() and c.m <= 16:
            for idx, label in enumerate(c.labels):
                if word is not None and idx == word.code:
                    continue
                ax.annotate(label, (idx, level[idx]), xytext=(0, 6),
                            textcoords="offset points", ha="center",
                            fontsize=6, color="#333333")
        if word is not None:
            ax.scatter([word.code], [level[word.code]], s=110,
                       color="#d62728", edgecolor="#7f0000", zorder=4)
            ax.annotate(
                f"{word.word}\nk = {word.code}", (word.code, level[word.code]),
                xytext=(0, 12), textcoords="offset points", ha="center",
                fontsize=8, color="#b00020",
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec="#d62728", alpha=0.95))
        if c.m <= 16:
            ticks = list(range(c.m))
        else:
            step = max(1, c.m // 8)
            ticks = list(range(0, c.m, step))
            if ticks[-1] != c.m - 1:
                ticks.append(c.m - 1)
        ax.set_xticks(ticks)
        ax.set_xlim(-0.8, c.m - 0.2)
        ax.set_ylim(-1.3, 1.45)
        ax.set_xlabel("código PCM k (offset binario)")
        ax.set_ylabel("nivel PAM normalizado")
        title = (f"Constelación PCM/PAM · M = {c.m} niveles · "
                 f"{c.bits_per_symbol} bits por muestra")
        if word is not None:
            title += f" · {word.word} → k = {word.code}"
        ax.set_title(title)

    def _draw_word_bits(self, ax, c, word) -> None:
        if word is None:
            ax.axis("off")
            ax.text(0.5, 0.5,
                    f"Escriba una palabra de {c.bits_per_symbol} bits (0/1).",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="#b00020")
            return
        bits = np.array([int(ch) for ch in word.word], dtype=np.float64)
        x = np.arange(bits.size + 1, dtype=np.float64)
        y = np.concatenate((bits, bits[-1:]))
        ax.step(x, y, where="post", color="#2ca02c", lw=1.6)
        for i, bit in enumerate(bits):
            ax.axvline(float(i), color="#bbbbbb", lw=0.6, ls=":")
            ax.text(float(i) + 0.5, 1.08, str(int(bit)), ha="center",
                    va="bottom", fontsize=8, color="#333333")
        ax.set_xlim(0.0, float(bits.size))
        ax.set_ylim(-0.2, 1.3)
        ax.set_yticks([0, 1])
        ax.set_xlabel("bit en serie (MSB primero)")
        ax.set_ylabel("bit")
        ax.set_title(
            f"Palabra PCM {word.word} = {word.code} · {word.bits} bits · "
            f"cada bit ocupa Tb = Ts/{word.bits}")

    def _draw_iq(self, ax, c) -> None:
        pts = c.points
        x = np.real(pts)
        y = np.imag(pts)
        limit = max(1.2, 1.28 * float(np.max(np.abs(np.concatenate((x, y))))))
        ax.axhline(0.0, color="#999999", lw=0.7)
        ax.axvline(0.0, color="#999999", lw=0.7)
        if c.technique == "PSK":
            theta = np.linspace(0.0, 2.0 * np.pi, 400)
            ax.plot(np.cos(theta), np.sin(theta), ls="--", lw=0.8,
                    color="#bbbbbb")
        for k in range(c.m):
            ax.plot([0.0, x[k]], [0.0, y[k]], color="#c7c7c7", lw=0.6, zorder=1)
        ax.scatter(x, y, s=46, color="#1f77b4", edgecolor="#0b3d61",
                   zorder=3)
        if self.labels_var.get() and c.m <= 64:
            for k, label in enumerate(c.labels):
                ax.annotate(label, (x[k], y[k]), xytext=(4, 4),
                            textcoords="offset points", fontsize=7,
                            color="#333333")
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("I (en fase)")
        ax.set_ylabel("Q (cuadratura)")
        ax.set_title(
            f"Constelación {c.technique} · M = {c.m} · "
            f"{c.bits_per_symbol} bit(s)/símbolo · "
            f"d_min = {c.min_distance:.3f}")

    def _draw_fsk(self, ax, c) -> None:
        f = c.frequencies
        ax.axhline(0.0, color="#888888", lw=0.7)
        for k, freq in enumerate(f):
            ax.plot([freq, freq], [0.0, 1.0], color="#2ca02c", lw=1.2)
            ax.plot([freq], [1.0], "o", color="#1f77b4", ms=6)
            if self.labels_var.get() and c.m <= 32:
                ax.text(freq, 1.06, c.labels[k], ha="center", va="bottom",
                        fontsize=7, rotation=90, color="#333333")
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(0.0, 1.55)
        ax.set_yticks([])
        ax.set_xlabel("Frecuencia normalizada (f − fc)/Δf")
        ax.set_title(
            f"Constelación {c.technique} · M = {c.m} tonos · "
            f"{c.bits_per_symbol} bit(s)/símbolo")

    # ------------------------------------------------------------------
    def _clear_table(self) -> None:
        for item in self.table.get_children():
            self.table.delete(item)

    def _fill_table(self, c, word) -> None:
        self._clear_table()
        if c.technique == "PCM (PAM)":
            headings = ("código k", "palabra", "nivel PAM", "—",
                        "|nivel|", "—")
            widths = (80, 85, 100, 60, 85, 60)
        elif c.is_iq:
            headings = ("k", "bits", "I", "Q", "amplitud", "fase [°]")
            widths = (55, 85, 95, 95, 95, 95)
        else:
            headings = ("k", "bits", "frecuencia", "—", "—", "—")
            widths = (55, 85, 110, 70, 70, 70)
        for col, text, width in zip(self.columns, headings, widths):
            self.table.heading(col, text=text)
            self.table.column(col, width=width, anchor="center", stretch=False)
        if c.technique == "PCM (PAM)":
            rows = [(str(k), label, f"{level:+.4f}", "—",
                     f"{abs(level):.4f}", "—")
                    for k, (label, level) in enumerate(zip(c.labels, c.i_values))]
        else:
            rows = c.rows()
        selected = None
        for row in rows:
            item = self.table.insert("", "end", values=row)
            if (c.technique == "PCM (PAM)" and word is not None
                    and int(row[0]) == word.code):
                selected = item
                self.table.item(item, tags=("selected",))
        if selected is not None:
            self.table.see(selected)

    def _export_csv(self) -> None:
        try:
            c = constellation(self.technique_var.get(), int(self.bits_var.get()))
        except (ConstellationError, ValueError):
            return
        if c.technique == "PCM (PAM)":
            header = ["codigo", "palabra", "nivel_pam", "q_adc", "voltios"]
            rows = []
            for k in range(c.m):
                w = pcm_word(k, c.bits_per_symbol, self._vref())
                rows.append((str(w.code), w.word, f"{w.level:+.6f}",
                             f"{w.signed_code:+d}", f"{w.volts:+.6f}"))
        elif c.is_iq:
            header = ["simbolo", "bits", "I", "Q", "amplitud", "fase_deg"]
            rows = c.rows()
        else:
            header = ["simbolo", "bits", "frecuencia_normalizada"]
            rows = [(r[0], r[1], r[2]) for r in c.rows()]
        path = save_rows_csv(self, f"constelacion_{c.technique}",
                             header, rows)
        if path:
            self.app.set_status(f"CSV de constelación exportado: {path}")
