"""Bildschirmfoto des spiels: welt + spieler-HUD, OHNE die dev-oberflaeche.

`F12` oder der knopf in der dev-oberflaeche setzen nur einen WUNSCH
(`request()`). Eingeloest wird er in `runtime/loop.py` an genau einer stelle:
nach `ui_root.render()` und vor `devui.render()` -- dort steht im
framebuffer die welt samt HUD, aber noch kein ImGui-pixel. Der knopf in der
dev-oberflaeche wird in `devui.build()` geklickt, also ebenfalls vor dieser
stelle; sein foto entsteht damit noch im selben frame.

Auf dem haupt-thread bleibt nur das AUSLESEN (`glReadPixels`, muss im
GL-thread laufen). Umdrehen, PNG-kodieren und schreiben uebernimmt ein
arbeiter-thread, damit der frame nicht hakt (`_write_png`).
"""
import os
import struct
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np


class ScreenshotWriter:
    """Haelt den wunsch, liest den framebuffer und schreibt im hintergrund."""

    def __init__(self, directory):
        self.directory = str(directory)
        self._pending = False
        self._executor = None
        self._futures = []
        # Vergebene namen: eine datei, die der arbeiter noch nicht
        # geschrieben hat, sieht os.path.exists nicht.
        self._issued = set()
        #: Pfad des zuletzt ANGESTOSSENEN fotos (die datei kann noch fehlen,
        #: solange der arbeiter schreibt).
        self.last_path = None

    def request(self):
        self._pending = True

    @property
    def pending(self):
        return self._pending

    def _next_path(self):
        stamp = time.strftime('%Y-%m-%d_%H%M%S')
        base = os.path.join(self.directory, f'spacesim_{stamp}')
        path = base + '.png'
        # Zwei fotos in derselben sekunde duerfen sich nicht ueberschreiben.
        n = 2
        while os.path.exists(path) or path in self._issued:
            path = f'{base}_{n}.png'
            n += 1
        self._issued.add(path)
        return path

    def capture_if_requested(self, gl_ctx, width, height):
        """Liest den framebuffer, falls ein wunsch ansteht. Liefert den pfad.

        `viewport=` explizit: `ctx.screen` kennt nach einem resize seine
        groesse nicht mehr (`.claude/rules/rendering.md`).
        """
        if not self._pending:
            return None
        self._pending = False
        w = max(1, int(width))
        h = max(1, int(height))
        data = gl_ctx.screen.read(viewport=(0, 0, w, h), components=3,
                                  alignment=1, dtype='f1')
        path = self._next_path()
        self.last_path = path
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix='screenshot')
        self._futures = [f for f in self._futures if not f.done()]
        self._futures.append(self._executor.submit(_write_png, data, w, h, path))
        return path

    def wait(self, timeout=None):
        """Bis alle angestossenen fotos geschrieben sind."""
        for future in list(self._futures):
            future.result(timeout=timeout)
        self._futures = []

    def shutdown(self):
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        self._futures = []


def _write_png(data, width, height, path):
    """Auf dem arbeiter: zeilen umdrehen (GL liest von unten), PNG schreiben.

    BEWUSST nicht `pygame.image.save`: das haelt den GIL fuer die ganze
    kodierung, und der haupt-thread stand dabei gemessen bis zu 520 ms still
    (2560x1440) -- genau der haenger, den der arbeiter vermeiden soll.
    `zlib.compress` gibt den GIL waehrend des komprimierens frei; der rest
    sind ein numpy-umdrehen und ein paar bytes chunk-rahmen.
    """
    try:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        rows = np.frombuffer(data, dtype=np.uint8).reshape(height, width * 3)
        raw = np.empty((height, width * 3 + 1), dtype=np.uint8)
        raw[:, 0] = 0                    # filter "none" je zeile
        raw[:, 1:] = rows[::-1]
        png = (b'\x89PNG\r\n\x1a\n'
               + _chunk(b'IHDR', struct.pack('>IIBBBBB', width, height,
                                             8, 2, 0, 0, 0))
               + _chunk(b'IDAT', zlib.compress(raw, 6))
               + _chunk(b'IEND', b''))
        with open(path, 'wb') as fh:
            fh.write(png)
        print(f"SCREENSHOT: {path} ({width}x{height})", flush=True)
    except Exception as exc:
        print(f"SCREENSHOT: fehlgeschlagen ({path}): {exc}", flush=True)
        raise


def _chunk(kind, payload):
    return (struct.pack('>I', len(payload)) + kind + payload
            + struct.pack('>I', zlib.crc32(kind + payload) & 0xFFFFFFFF))
