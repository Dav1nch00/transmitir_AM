"""Utilidades compartidas por los módulos de compresión del Desafío 4."""
from __future__ import annotations


def size_change(original: int, compressed: int) -> str:
    """Describe el cambio de tamaño de forma inequívoca.

    Un porcentaje negativo significa que el archivo **creció**: el ratio es
    menor que 1 y el texto lo indica diciendo «archivo X % más grande».
    """
    if original <= 0 or compressed <= 0:
        return "n/d (tamaños no disponibles)"
    ratio = original / compressed
    delta = 100.0 * (1.0 - compressed / original)
    if delta >= 0.0:
        return f"{ratio:.2f}×  ({delta:.1f} % más pequeño)"
    return f"{ratio:.2f}×  (archivo {abs(delta):.1f} % más grande)"
