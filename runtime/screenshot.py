"""Bildschirmfoto des spiels: welt + spieler-HUD, OHNE die dev-oberflaeche.

`F12` oder der knopf in der dev-oberflaeche setzen nur einen WUNSCH
(`request()`). Eingeloest wird er in `runtime/loop.py`, je nach aufloesung an
einer von zwei stellen:

- FENSTERAUFLOESUNG (`window`): nach `ui_root.render()` und vor
  `devui.render()` -- dort steht im framebuffer die welt samt HUD, aber noch
  kein ImGui-pixel. Der knopf der dev-oberflaeche wird in `devui.build()`
  geklickt, also ebenfalls vor dieser stelle.
- 2K/4K/6K/8K (`render_scaled`): VOR dem normalen `renderer.render()` wird
  derselbe frame ein zweites mal in einen offscreen-framebuffer gezeichnet,
  in der zielhoehe. Davor, nicht danach: der renderer veroeffentlicht beim
  zeichnen schirmpositionen (`apsis_marker_hits`, ...), und die naechsten
  klicks muessen die des FENSTERS sehen. Ein knopfdruck in der
  dev-oberflaeche landet deshalb erst im folgenden frame.

ZUSCHNITT: nur die BREITE ist waehlbar (linker/rechter rand als anteil der
fensterbreite), die hoehe ist immer die volle -- die stufen 2K..8K SIND
hoehen. Die ausgabebreite folgt daraus: zuschnittbreite x zielhoehe /
fensterhoehe.

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


#: (beschriftung, ausgabehoehe in px). None = fensteraufloesung, ohne
#: neuzeichnen. 6K als 16:9-hoehe (5760x3240), passend zu 4K/8K.
RESOLUTIONS = (
    ('window', None),
    ('2K', 1440),
    ('4K', 2160),
    ('5K', 2700),
    ('5.5K', 2970),
    ('6K', 3240),
    ('8K', 4320),
    ('12K', 6480),
    ('7K', 3780),
    ('10K', 5400),
    ('16K', 8640),
)

#: Schmalster zuschnitt, als anteil der fensterbreite.
CROP_MIN_FRACTION = 0.05
#: Randlinie des zuschnitts in ECHTEN pixeln -- bewusst fest und nicht in
#: config.json: sie ist ein werkzeug-strich, kein teil des HUDs.
CROP_BORDER_PX = 2.0
#: Abdunklung ausserhalb des zuschnitts und farbe der randlinie.
CROP_SHADE = (0.0, 0.0, 0.0, 0.55)
CROP_BORDER_COLOR = (1.0, 1.0, 1.0, 0.85)


class ScreenshotWriter:
    """Haelt wunsch und einstellungen, liest aus und schreibt im hintergrund."""

    def __init__(self, directory, resolution='window'):
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
        #: Groesse des zuletzt angestossenen fotos (breite, hoehe).
        self.last_size = None
        #: Index in RESOLUTIONS.
        self.resolution_index = 0
        self.set_resolution(resolution)
        #: Zuschnitt als anteil der fensterbreite, 0..1. Vorgabe: alles.
        self.crop_left = 0.0
        self.crop_right = 1.0
        #: Beide raender gespiegelt um die mitte bewegen.
        self.crop_symmetric = False
        #: Abdunklung + randlinien im fenster zeigen (nur bei echtem
        #: zuschnitt -- ohne gibt es nichts zu zeigen).
        self.show_crop_overlay = True

    def request(self):
        self._pending = True

    @property
    def pending(self):
        return self._pending

    # ------------------------------------------------------- einstellungen

    def set_resolution(self, value):
        """Per index oder beschriftung ('window', '2K', ...)."""
        if isinstance(value, str):
            labels = [label.lower() for label, _h in RESOLUTIONS]
            key = value.strip().lower()
            if key not in labels:
                raise ValueError(f"unbekannte aufloesung {value!r}")
            value = labels.index(key)
        self.resolution_index = max(0, min(len(RESOLUTIONS) - 1, int(value)))

    @property
    def resolution_label(self):
        return RESOLUTIONS[self.resolution_index][0]

    def output_height(self, window_height):
        target = RESOLUTIONS[self.resolution_index][1]
        return int(window_height) if target is None else int(target)

    def needs_rerender(self, window_height):
        """True, wenn das foto NICHT einfach der fensterinhalt ist."""
        return self.output_height(window_height) != int(window_height)

    def crop_pixels(self, full_width):
        """(x0, breite) des zuschnitts auf einem bild der breite `full_width`."""
        full_width = max(1, int(full_width))
        x0 = int(round(self.crop_left * full_width))
        x1 = int(round(self.crop_right * full_width))
        x0 = max(0, min(full_width - 1, x0))
        x1 = max(x0 + 1, min(full_width, x1))
        return x0, x1 - x0

    def scaled_frame_size(self, window_width, window_height):
        """Groesse des ganzen, neu gezeichneten bildes (vor dem zuschnitt)."""
        out_h = self.output_height(window_height)
        k = out_h / max(1.0, float(window_height))
        return max(1, int(round(window_width * k))), out_h

    def output_size(self, window_width, window_height):
        """Groesse der PNG-datei: (zuschnittbreite, zielhoehe)."""
        full_w, out_h = self.scaled_frame_size(window_width, window_height)
        return self.crop_pixels(full_w)[1], out_h

    @property
    def crop_is_full(self):
        return self.crop_left <= 1e-6 and self.crop_right >= 1.0 - 1e-6

    def set_crop(self, left, right):
        """Beide raender setzen; geklemmt auf 0..1 und mindestbreite."""
        left = max(0.0, min(1.0 - CROP_MIN_FRACTION, float(left)))
        right = max(left + CROP_MIN_FRACTION, min(1.0, float(right)))
        self.crop_left, self.crop_right = left, right

    def set_crop_edge(self, edge, value):
        """Einen rand ziehen. Symmetrisch: der andere spiegelt um die mitte."""
        value = float(value)
        if edge == 'left':
            value = max(0.0, min(0.5 - CROP_MIN_FRACTION * 0.5, value)) \
                if self.crop_symmetric else value
            other = 1.0 - value if self.crop_symmetric else self.crop_right
            self.set_crop(value, other)
        elif edge == 'right':
            value = max(0.5 + CROP_MIN_FRACTION * 0.5, min(1.0, value)) \
                if self.crop_symmetric else value
            other = 1.0 - value if self.crop_symmetric else self.crop_left
            self.set_crop(other, value)
        else:
            raise ValueError(edge)

    def centre_crop(self):
        """Den zuschnitt bei gleicher breite auf die bildmitte schieben."""
        width = self.crop_right - self.crop_left
        self.set_crop(0.5 - width * 0.5, 0.5 + width * 0.5)

    def reset_crop(self):
        self.set_crop(0.0, 1.0)

    # --------------------------------------------------------------- overlay

    def draw_crop_overlay(self, ui):
        """Abdunklung ausserhalb des zuschnitts + die zwei randlinien.

        `ui` ist der UIContext des spieler-HUDs; gezeichnet wird NACH dem
        foto (runtime/loop.py), es landet also nie selbst im bild.
        """
        if not self.show_crop_overlay or self.crop_is_full:
            return
        width, height = float(ui.width), float(ui.height)
        x0, cw = self.crop_pixels(ui.width)
        x1 = float(x0 + cw)
        x0 = float(x0)
        if x0 > 0.0:
            ui.draw.rect(0.0, 0.0, x0, height, fill=CROP_SHADE)
        if x1 < width:
            ui.draw.rect(x1, 0.0, width - x1, height, fill=CROP_SHADE)
        line = CROP_BORDER_PX
        # Die linien liegen AUSSERHALB des zuschnitts: was zwischen ihnen
        # steht, ist genau das foto.
        if x0 > 0.0:
            ui.draw.rect(max(0.0, x0 - line), 0.0, line, height,
                         fill=CROP_BORDER_COLOR)
        if x1 < width:
            ui.draw.rect(x1, 0.0, line, height, fill=CROP_BORDER_COLOR)
        ui.draw.flush()

    # --------------------------------------------------------------- foto

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

    def _submit(self, data, width, height):
        path = self._next_path()
        self.last_path = path
        self.last_size = (int(width), int(height))
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix='screenshot')
        self._futures = [f for f in self._futures if not f.done()]
        self._futures.append(
            self._executor.submit(_write_png, data, width, height, path))
        return path

    def capture_if_requested(self, gl_ctx, width, height):
        """Fensteraufloesung: liest den zuschnitt aus dem framebuffer.

        Tut nichts, solange kein wunsch ansteht ODER eine hoehere aufloesung
        gewaehlt ist (die zieht `render_scaled` vor dem normalen zeichnen).
        `viewport=` explizit: `ctx.screen` kennt nach einem resize seine
        groesse nicht mehr (`.claude/rules/rendering.md`).
        """
        if not self._pending or self.needs_rerender(height):
            return None
        self._pending = False
        w = max(1, int(width))
        h = max(1, int(height))
        x0, cw = self.crop_pixels(w)
        data = gl_ctx.screen.read(viewport=(x0, 0, cw, h), components=3,
                                  alignment=1, dtype='f1')
        return self._submit(data, cw, h)

    def render_scaled(self, app, draw_frame):
        """2K..8K: den frame offscreen in der zielhoehe neu zeichnen.

        Alles, was an der fenstergroesse haengt, wird fuer diesen einen
        zeichengang auf das k-fache gestellt (k = zielhoehe / fensterhoehe)
        und danach EXAKT zurueckgesetzt: renderer (viewport, FXAA-ziele,
        fonts), kamera (breite, hoehe, massstab -- derselbe weltausschnitt),
        spieler-HUD (groesse, ui_scale, mausposition fuer den hover). Die
        ui_scale wird k-fach GESETZT, nicht aus der hoehe abgeleitet: die
        klemme `ui_scale_max` wuerde das HUD sonst im 8K-bild relativ
        schrumpfen lassen. Liniendicken folgen ueber `capture_scale`.

        `draw_frame()` zeichnet welt + HUD (runtime/loop.py, derselbe aufruf
        wie im normalen frame, mit real_dt = 0). Liefert den pfad oder None.
        """
        win_h = int(app.renderer.height)
        if not self._pending or not self.needs_rerender(win_h):
            return None
        self._pending = False
        renderer = app.renderer
        camera = app.camera
        ui_root = app.ui_root
        gl_ctx = app.window.ctx
        win_w = int(renderer.width)
        full_w, out_h = self.scaled_frame_size(win_w, win_h)
        limit = _max_framebuffer_size(gl_ctx)
        if full_w > limit or out_h > limit:
            print(f"SCREENSHOT: {full_w}x{out_h} uebersteigt die GL-grenze "
                  f"{limit} px -- abgebrochen", flush=True)
            return None
        k = out_h / float(win_h)

        saved = {
            'cam': (camera.width, camera.height, camera.scale,
                    camera.target_scale),
            'ui_scale': renderer.ui_scale,
            'mouse': getattr(ui_root, '_mouse_pos', None),
        }
        texture = gl_ctx.texture((full_w, out_h), 4)
        fbo = gl_ctx.framebuffer(color_attachments=[texture])
        t0 = time.perf_counter()
        try:
            renderer.resize(full_w, out_h, ui_scale=saved['ui_scale'] * k)
            renderer.capture_scale = k
            renderer.output_framebuffer = fbo
            camera.width, camera.height = full_w, out_h
            camera.scale = saved['cam'][2] * k
            camera.target_scale = saved['cam'][3] * k
            ui_root.resize(full_w, out_h, ui_scale=renderer.ui_scale)
            if saved['mouse'] is not None:
                ui_root._mouse_pos = (saved['mouse'][0] * k,
                                      saved['mouse'][1] * k)
            fbo.use()
            if app.hud is not None:
                app.hud.update()
            ui_root.begin_frame(0.0)
            draw_frame()
            x0, cw = self.crop_pixels(full_w)
            data = fbo.read(viewport=(x0, 0, cw, out_h), components=3,
                            alignment=1, dtype='f1')
        except Exception as exc:
            # Ein misslungenes foto darf das spiel nicht beenden; der
            # zustand wird unten trotzdem zurueckgesetzt.
            print(f"SCREENSHOT: neuzeichnen fehlgeschlagen ({exc})", flush=True)
            data = None
        finally:
            renderer.output_framebuffer = None
            renderer.capture_scale = 1.0
            # resize() ohne ui_scale leitet sie wieder aus der fensterhoehe
            # ab -- derselbe wert wie vorher, solange niemand den regler
            # angefasst hat (und das kann waehrend des fotos keiner).
            renderer.resize(win_w, win_h)
            (camera.width, camera.height, camera.scale,
             camera.target_scale) = saved['cam']
            ui_root.resize(win_w, win_h, ui_scale=renderer.ui_scale)
            if saved['mouse'] is not None:
                ui_root._mouse_pos = saved['mouse']
            gl_ctx.screen.use()
            if app.hud is not None:
                app.hud.update()
            ui_root.begin_frame(0.0)
            fbo.release()
            texture.release()
        if data is None:
            return None
        render_ms = (time.perf_counter() - t0) * 1000.0
        path = self._submit(data, cw, out_h)
        print(f"SCREENSHOT: {self.resolution_label} neu gezeichnet "
              f"({full_w}x{out_h}, zuschnitt {cw} px) in {render_ms:.0f} ms",
              flush=True)
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


def _max_framebuffer_size(gl_ctx):
    """Kleinste der GL-grenzen fuer textur und viewport."""
    info = getattr(gl_ctx, 'info', {}) or {}
    limit = int(info.get('GL_MAX_TEXTURE_SIZE', 16384) or 16384)
    dims = info.get('GL_MAX_VIEWPORT_DIMS')
    if dims:
        try:
            limit = min(limit, int(min(dims)))
        except (TypeError, ValueError):
            pass
    return limit


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
