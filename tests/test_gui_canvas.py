"""Regresión: el canvas de cada pestaña debe ser hijo de su plot_frame.

Si el canvas se crea con ``master=self`` y se empaqueta con ``in_=plot_frame``,
el frame queda dibujado encima y las gráficas no se ven aunque Matplotlib
genere correctamente el buffer.
"""
from __future__ import annotations

import tkinter as tk
import unittest

from gui.app import App


class TestCanvasHierarchy(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.app = App()
            cls.app.withdraw()
        except tk.TclError as exc:  # pragma: no cover - entorno sin pantalla
            raise unittest.SkipTest(f"Sin servidor gráfico: {exc}") from exc

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "app"):
            cls.app.destroy()

    def test_every_canvas_belongs_to_its_plot_frame(self) -> None:
        self.assertEqual(len(self.app.tabs), 11)
        for tab in self.app.tabs:
            with self.subTest(tab=type(tab).__name__):
                canvas = tab.canvas.get_tk_widget()
                self.assertEqual(canvas.winfo_parent(), str(tab.plot_frame))

    def test_draw_sync_does_not_raise(self) -> None:
        for tab in self.app.tabs:
            with self.subTest(tab=type(tab).__name__):
                tab.refresh()


if __name__ == "__main__":
    unittest.main()
