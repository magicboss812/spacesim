"""Die vorhersagelinie im pixel-stil -- fuenf zeichenweisen plus die alte.

`renderer.prediction_line_style` waehlt:

    klassisch   die alte glatte GL-linie (farbe und deckkraft einstellbar)
    kontur      2-px-pixelband mit dunklem 1-px-saum
    perlen      2x2-zellen im zeittakt auf einem duennen faden
    lineal      1-px-linie mit zeitstrichen, jeder n-te lang
    halbton     deckung faellt mit der vorausschau (geordnetes dithern)
    schiene     3-px-zweiton-band mit pfeilspitzen in flugrichtung

Zwei paesse, wie beim schiff:

1. **Puffer** in fenstergroesse, blending MAX. Jedes segment der linie ist
   ein instanziertes quad; sein fragment-shader misst den abstand der
   PIXELMITTE zur strecke und schreibt kanalweise: r = linienfarbe, g =
   dunkler saum, b = schiene (dunklere linienfarbe), a = markenfarbe. Die
   marken (perlen, striche, pfeile) sind eigene quads in denselben puffer.
   MAX statt ueberblenden, weil sich benachbarte segmente an jeder fuge
   ueberlappen: mit alpha-blending lagen dort dunklere flecken (saum ueber
   saum) und hellere (kern ueber kern).
2. **Zusammenbau** ueber die 16-px-kacheln, die die linie beruehrt: die
   vier kanaele in der reihenfolge saum -> schiene -> linie -> marke auf den
   frame. Danach werden dieselben kacheln im puffer geloescht. Ueber das
   ganze rechteck der linie (bei naher kamera fast der schirm) kostete
   allein dieser pass in software-GL mehrere ms.

Zeit je punkt: die gezeichneten runs tragen keine zeit. Die projizierte
dichte kurve (`_pred_time_src`, aus `_adaptive_prediction_screen_points`)
tut es; `_map_run_times` laeuft monoton an ihr entlang. Marken sitzen auf
VIELFACHEN der absoluten zeit, nicht auf abstaenden vom schiff: so stehen
sie auf der bahn still und das schiff faehrt darueber. Vom schiff aus
gezaehlt wanderten sie jedes frame die linie entlang.

Alle breiten und abstaende sind DESIGN-EINHEITEN (`ui_px`), gerundet auf
ganze pixel.
"""
import moderngl
import numpy as np

try:
    from numba import njit
    _NUMBA = True
except Exception:  # pragma: no cover - ohne numba laeuft dieselbe funktion in python
    _NUMBA = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f


STYLES = ("klassisch", "kontur", "perlen", "lineal", "halbton", "schiene")
_STYLE_CODE = {"kontur": 1, "perlen": 2, "lineal": 3, "halbton": 4, "schiene": 5}

#: Vorgaben je zeichenweise. `renderer.prediction_line_styles` (config, dev-UI)
#: ueberschreibt einzelne werte; was fehlt, kommt von hier.
DEFAULT_STYLE_SETTINGS = {
    "klassisch": {"width": 2.0, "alpha": 0.6},
    "kontur": {"width": 2.0, "outline": 1.0, "outline_alpha": 0.70},
    "perlen": {"size": 2.0, "spacing": 9.0, "major_every": 6,
               "thread_alpha": 0.38, "ring_alpha": 0.5},
    "lineal": {"width": 1.0, "tick": 3.0, "major_tick": 4.0, "spacing": 7.0,
               "major_every": 5},
    "halbton": {"width": 2.0, "min_coverage": 0.22, "steps": 5},
    "schiene": {"width": 3.0, "arrow": 4.0, "spacing": 22.0,
                "rail_alpha": 0.80, "outline_alpha": 0.55},
}

DEFAULT_LINE_COLOR = "#97dce4"     # HUD-cyan (#17b2c4), zum weiss hin lesbar gemacht
DEFAULT_MARK_COLOR = "#e6f4f8"
_GROUND = (0.016, 0.027, 0.047)    # HUD-grund #04070c: der saum

# Zeitraster der marken (s): runde werte von 1 s bis 100 jahre.
_DT_LADDER = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200,
              10800, 21600, 43200, 86400, 172800, 432000, 864000, 1728000,
              2592000, 5184000, 8640000, 15552000, 31536000, 63072000,
              157680000, 315360000, 788400000, 1576800000, 3153600000]
_MAX_MARKS = 1500
_ANCHOR_SPACING_PX = 24.0          # ankertakt fuer das dithern ohne marken
_SEG_FLOATS = 8
_TILE = 16


def _hex_rgb(value, fallback):
    try:
        text = str(value).strip().lstrip('#')
        if len(text) == 3:
            text = ''.join(ch * 2 for ch in text)
        return tuple(int(text[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    except Exception:
        return _hex_rgb(fallback, "#ffffff") if fallback != value else (1.0, 1.0, 1.0)


@njit(cache=True)
def _map_run_times(rx, ry, starts, ends, sx, sy, st, tol2):
    """Zeit je run-punkt: monoton an der dichten projizierten kurve entlang.

    Jeder run beginnt beim ERSTEN quellsegment ab dem cursor, das naeher als
    sqrt(tol2) px liegt -- nicht beim naechsten ueberhaupt: eine bahn ueber
    mehrere umlaeufe kreuzt sich selbst, und der naechste treffer koennte ein
    spaeterer umlauf sein. Danach rueckt der cursor nur vor, solange das
    folgende segment naeher liegt."""
    n = rx.shape[0]
    m = sx.shape[0]
    out = np.zeros(n)
    if m < 2:
        return out
    j = 0
    for r in range(starts.shape[0]):
        a = starts[r]
        b = ends[r]
        if b <= a:
            continue
        px = rx[a]
        py = ry[a]
        best = j
        bestd = 1e300
        for k in range(j, m - 1):
            dx = sx[k + 1] - sx[k]
            dy = sy[k + 1] - sy[k]
            l2 = dx * dx + dy * dy
            s = 0.0
            if l2 > 1e-12:
                s = ((px - sx[k]) * dx + (py - sy[k]) * dy) / l2
                s = min(1.0, max(0.0, s))
            ex = sx[k] + dx * s - px
            ey = sy[k] + dy * s - py
            d = ex * ex + ey * ey
            if d < bestd:
                bestd = d
                best = k
            if d < tol2:
                best = k
                break
        j = best
        for i in range(a, b):
            px = rx[i]
            py = ry[i]
            while True:
                dx = sx[j + 1] - sx[j]
                dy = sy[j + 1] - sy[j]
                l2 = dx * dx + dy * dy
                s = 0.0
                if l2 > 1e-12:
                    s = ((px - sx[j]) * dx + (py - sy[j]) * dy) / l2
                    s = min(1.0, max(0.0, s))
                ex = sx[j] + dx * s - px
                ey = sy[j] + dy * s - py
                d = ex * ex + ey * ey
                if j + 2 >= m:
                    break
                dx2 = sx[j + 2] - sx[j + 1]
                dy2 = sy[j + 2] - sy[j + 1]
                l22 = dx2 * dx2 + dy2 * dy2
                s2 = 0.0
                if l22 > 1e-12:
                    s2 = ((px - sx[j + 1]) * dx2 + (py - sy[j + 1]) * dy2) / l22
                    s2 = min(1.0, max(0.0, s2))
                ex2 = sx[j + 1] + dx2 * s2 - px
                ey2 = sy[j + 1] + dy2 * s2 - py
                if ex2 * ex2 + ey2 * ey2 <= d:
                    j += 1
                else:
                    break
            out[i] = st[j] + (st[j + 1] - st[j]) * s
    return out


def _pick_dt(px_per_s, min_px, current):
    """Kleinster wert der leiter mit abstand >= min_px. Der laufende wert
    bleibt, solange er zwischen 0.85 und 3.2 x min_px liegt -- die leiter
    springt um hoechstens 2.5x, also gibt es keinen wert, an dem zwei stufen
    abwechselnd gewinnen."""
    if px_per_s <= 0.0:
        return None
    if current is not None:
        gap = current * px_per_s
        if 0.85 * min_px <= gap <= 3.2 * min_px:
            return current
    for dt in _DT_LADDER:
        if dt * px_per_s >= min_px:
            return float(dt)
    return float(_DT_LADDER[-1])


class PredictionPixelMixin:
    """Zeichnet die vorhersagelinie in der gewaehlten zeichenweise."""

    # ---- einstellungen ----------------------------------------------------

    def _pred_style(self):
        style = str(getattr(self, 'prediction_line_style', 'kontur'))
        return style if style in STYLES else 'kontur'

    def _pred_settings(self, style=None):
        style = style or self._pred_style()
        merged = dict(DEFAULT_STYLE_SETTINGS.get(style, {}))
        own = (getattr(self, 'prediction_line_styles', None) or {}).get(style) or {}
        for key, value in own.items():
            if key in merged:
                try:
                    merged[key] = type(merged[key])(value)
                except Exception:
                    pass
        return merged

    def _pred_colors(self):
        line = _hex_rgb(getattr(self, 'prediction_line_color', DEFAULT_LINE_COLOR),
                        DEFAULT_LINE_COLOR)
        mark = _hex_rgb(getattr(self, 'prediction_line_mark_color', DEFAULT_MARK_COLOR),
                        DEFAULT_MARK_COLOR)
        return line, mark

    def _pred_px(self, design_units, minimum=1):
        try:
            value = float(self.ui_px(float(design_units)))
        except Exception:
            value = float(design_units)
        return max(int(minimum), int(round(value)))

    def classic_prediction_line(self):
        """(farbe rgba, breite) der alten linie -- auch fuer `klassisch`."""
        s = self._pred_settings('klassisch')
        line, _mark = self._pred_colors()
        return (line[0], line[1], line[2], float(s['alpha'])), float(s['width'])

    # ---- GL ---------------------------------------------------------------

    def _init_prediction_pixel_pipeline(self):
        self._pred_seg_program = self._compile_shader_program(
            'pred_seg.vert', 'pred_seg.frag', 'pred_seg')
        self._pred_mark_program = self._compile_shader_program(
            'pred_mark.vert', 'pred_mark.frag', 'pred_mark')
        self._pred_compose_program = self._compile_shader_program(
            'pred_tile.vert', 'pred_compose.frag', 'pred_compose')
        self._pred_clear_program = self._compile_shader_program(
            'pred_tile.vert', 'pred_clear.frag', 'pred_clear')
        self._pred_corner_vbo = None
        self._pred_seg_vbo = None
        self._pred_mark_vbo = None
        self._pred_seg_vao = None
        self._pred_mark_vao = None
        self._pred_compose_vao = None
        self._pred_clear_vao = None
        self._pred_tile_vbo = None
        self._pred_tile_cap = 0
        self._pred_seg_cap = 0
        self._pred_mark_cap = 0
        self._pred_buf_tex = None
        self._pred_buf_fbo = None
        self._pred_buf_size = None
        if None in (self._pred_seg_program, self._pred_mark_program,
                    self._pred_compose_program, self._pred_clear_program):
            return
        try:
            corners = np.array([0.0, -1.0, 1.0, -1.0, 0.0, 1.0, 1.0, 1.0], dtype='f4')
            self._pred_corner_vbo = self.ctx.buffer(corners.tobytes())
            self._pred_compose_program['u_buf'].value = 0
            self._pred_ensure_tiles(256)
        except Exception as exc:
            self.debug_info['shader_error'] = f"pred_pixel: {exc}"
            self._pred_corner_vbo = None
            self._pred_compose_vao = None

    def _pred_pixel_ready(self):
        return (getattr(self, '_pred_compose_vao', None) is not None
                and getattr(self, '_pred_corner_vbo', None) is not None)

    def _pred_ensure_instances(self, kind, count):
        """Instanzpuffer, der nur waechst (kein GL-objekt je frame)."""
        cap_attr = '_pred_%s_cap' % kind
        if count <= getattr(self, cap_attr):
            return
        cap = max(256, getattr(self, cap_attr))
        while cap < count:
            cap *= 2
        vbo_attr, vao_attr = '_pred_%s_vbo' % kind, '_pred_%s_vao' % kind
        for attr in (vao_attr, vbo_attr):
            obj = getattr(self, attr)
            if obj is not None:
                try:
                    obj.release()
                except Exception:
                    pass
        vbo = self.ctx.buffer(reserve=cap * _SEG_FLOATS * 4, dynamic=True)
        prog = self._pred_seg_program if kind == 'seg' else self._pred_mark_program
        vao = self.ctx.vertex_array(prog, [
            (self._pred_corner_vbo, '2f', 'a_corner'),
            (vbo, '4f 4f/i', 'i_a', 'i_b'),
        ])
        setattr(self, vbo_attr, vbo)
        setattr(self, vao_attr, vao)
        setattr(self, cap_attr, cap)

    def _pred_ensure_tiles(self, count):
        """Kachel-instanzen fuer zusammenbau und loeschen (ein puffer, zwei
        VAOs); waechst nur."""
        if count <= self._pred_tile_cap and self._pred_compose_vao is not None:
            return
        cap = max(256, self._pred_tile_cap)
        while cap < count:
            cap *= 2
        for attr in ('_pred_compose_vao', '_pred_clear_vao', '_pred_tile_vbo'):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    obj.release()
                except Exception:
                    pass
        vbo = self.ctx.buffer(reserve=cap * 2 * 4, dynamic=True)
        self._pred_tile_vbo = vbo
        self._pred_compose_vao = self.ctx.vertex_array(self._pred_compose_program, [
            (self._pred_corner_vbo, '2f', 'a_corner'), (vbo, '2f/i', 'i_tile')])
        self._pred_clear_vao = self.ctx.vertex_array(self._pred_clear_program, [
            (self._pred_corner_vbo, '2f', 'a_corner'), (vbo, '2f/i', 'i_tile')])
        self._pred_tile_cap = cap

    def _pred_tiles(self, segs, marks_gl, pad):
        """Linke untere ecken (GL-px) der kacheln, die linie oder marken
        beruehren. Lange segmente werden in halbe kacheln unterteilt, damit
        keine kachel dazwischen fehlt."""
        x0, y0, x1, y1 = segs[:, 0], segs[:, 1], segs[:, 2], segs[:, 3]
        L = np.hypot(x1 - x0, y1 - y0)
        n = np.maximum(1, np.ceil(L / (0.5 * _TILE)).astype(np.int64))
        idx = np.repeat(np.arange(len(segs)), n)
        start = np.repeat(np.cumsum(n) - n, n)
        f = (np.arange(idx.size) - start) / n[idx]
        px = np.concatenate((x0[idx] + (x1 - x0)[idx] * f, x1[-1:]))
        py = np.concatenate((y0[idx] + (y1 - y0)[idx] * f, y1[-1:]))
        if marks_gl is not None and len(marks_gl):
            px = np.concatenate((px, marks_gl[:, 0]))
            py = np.concatenate((py, marks_gl[:, 1]))
        W, H = int(self.width), int(self.height)
        nx, ny = (W + _TILE - 1) // _TILE, (H + _TILE - 1) // _TILE
        keys = []
        for ox in (-pad, pad):
            for oy in (-pad, pad):
                tx = np.floor((px + ox) / _TILE).astype(np.int64)
                ty = np.floor((py + oy) / _TILE).astype(np.int64)
                ok = (tx >= 0) & (tx < nx) & (ty >= 0) & (ty < ny)
                keys.append(ty[ok] * nx + tx[ok])
        keys = np.unique(np.concatenate(keys))
        return np.ascontiguousarray(np.column_stack(
            ((keys % nx) * _TILE, (keys // nx) * _TILE)), dtype='f4')

    def _pred_clear_all(self):
        try:
            if self._pred_buf_fbo is not None:
                self._pred_buf_fbo.clear(0.0, 0.0, 0.0, 0.0)
        except Exception:
            pass

    def _pred_ensure_buffer(self):
        w, h = int(self.width), int(self.height)
        if self._pred_buf_size == (w, h) and self._pred_buf_fbo is not None:
            return
        for obj in (self._pred_buf_fbo, self._pred_buf_tex):
            if obj is not None:
                try:
                    obj.release()
                except Exception:
                    pass
        tex = self.ctx.texture((w, h), 4)
        tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._pred_buf_tex = tex
        self._pred_buf_fbo = self.ctx.framebuffer(color_attachments=[tex])
        self._pred_buf_size = (w, h)
        self._pred_buf_fbo.clear(0.0, 0.0, 0.0, 0.0)

    # ---- geometrie --------------------------------------------------------

    def _pred_runs_with_time(self, runs, t_head, t_end):
        """Liste von (n, 4)-arrays: x, y (top-down px), t_abs, u (0 am schiff,
        1 am ende der gezeichneten linie)."""
        src = getattr(self, '_pred_time_src', None)
        runs = [np.asarray(r, dtype=np.float64).reshape(-1, 2) for r in runs if len(r) >= 2]
        if not runs:
            return []
        span = max(float(t_end) - float(t_head), 1e-9)
        allxy = np.concatenate(runs, axis=0)
        lens = np.array([len(r) for r in runs], dtype=np.int64)
        ends = np.cumsum(lens)
        starts = ends - lens
        if src is not None and len(src[0]) >= 2:
            sx, sy, st = src
            t = _map_run_times(np.ascontiguousarray(allxy[:, 0]),
                               np.ascontiguousarray(allxy[:, 1]),
                               starts, ends, sx, sy, st, 1.0)
        else:
            # Keine quelle (sollte nicht vorkommen): bogenlaenge als zeitersatz.
            seg = np.hypot(*np.diff(allxy, axis=0).T)
            cum = np.concatenate(([0.0], np.cumsum(seg)))
            t = float(t_head) + cum / max(cum[-1], 1e-9) * span
        out = []
        for a, b in zip(starts, ends):
            tt = np.maximum.accumulate(t[a:b])
            u = np.clip((tt - float(t_head)) / span, 0.0, 1.0)
            out.append(np.column_stack((allxy[a:b], tt, u)))
        return out

    def _pred_px_per_s(self, runs):
        """Bildschirmtempo der linie (px je sim-sekunde), 20-%-quantil:
        so haelt der grossteil der linie den mindestabstand der marken
        ein, auch dort, wo die bahn langsam ist (apoapsis)."""
        rates = []
        for r in runs:
            ds = np.hypot(np.diff(r[:, 0]), np.diff(r[:, 1]))
            dt = np.diff(r[:, 2])
            ok = (dt > 1e-9) & (ds > 1e-6)
            if ok.any():
                rates.append(ds[ok] / dt[ok])
        if not rates:
            return 0.0
        rates = np.concatenate(rates)
        return float(np.quantile(rates, 0.2))

    def _pred_crossings(self, runs, dt):
        """Punkte, an denen t_abs ein vielfaches von dt kreuzt: (x, y, tx, ty,
        u, k) mit k = index des vielfachen."""
        rows = []
        for r in runs:
            k = np.floor(r[:, 2] / dt)
            hit = np.flatnonzero(k[1:] > k[:-1])
            if hit.size == 0:
                continue
            i0, i1 = hit, hit + 1
            kk = k[i1]
            tm = kk * dt
            t0, t1 = r[i0, 2], r[i1, 2]
            f = np.clip((tm - t0) / np.maximum(t1 - t0, 1e-12), 0.0, 1.0)
            x = r[i0, 0] + (r[i1, 0] - r[i0, 0]) * f
            y = r[i0, 1] + (r[i1, 1] - r[i0, 1]) * f
            u = r[i0, 3] + (r[i1, 3] - r[i0, 3]) * f
            dx, dy = r[i1, 0] - r[i0, 0], r[i1, 1] - r[i0, 1]
            ln = np.maximum(np.hypot(dx, dy), 1e-9)
            rows.append(np.column_stack((x, y, dx / ln, dy / ln, u, kk)))
        return np.concatenate(rows, axis=0) if rows else np.zeros((0, 6))

    @staticmethod
    def _pred_thin(r, step=3.0):
        """Punkte im abstand von etwa `step` px. Die runs sind fuer die
        glatte linie bis unter einen pixel verfeinert (gemessen 0.58 px je
        segment bei 2e-5 px/m): als quads je segment schattierte damit jedes
        pixel rund zehnmal. Eine 3-px-sehne weicht bei 100 px kruemmungs-
        radius 0.011 px von der kurve ab."""
        if len(r) <= 2:
            return r
        seg = np.hypot(np.diff(r[:, 0]), np.diff(r[:, 1]))
        cum = np.concatenate(([0.0], np.cumsum(seg)))
        b = np.floor(cum / step)
        keep = np.flatnonzero(np.concatenate(([True], b[1:] != b[:-1])))
        if keep[-1] != len(r) - 1:
            keep = np.append(keep, len(r) - 1)
        return r[keep]

    def _pred_segments(self, runs, anchor_dt):
        """Instanzdaten je segment: p0, p1 (GL, y oben), u0, u1, a0, a1. `a`
        ist die bogenlaenge in px, verankert an einem vielfachen von
        `anchor_dt` -- damit steht das muster der dither auf der BAHN still."""
        h = float(self.height)
        parts = []
        for r in runs:
            r = self._pred_thin(r)
            x, y = r[:, 0], h - r[:, 1]
            seg = np.hypot(np.diff(x), np.diff(y))
            arc = np.concatenate(([0.0], np.cumsum(seg)))
            if anchor_dt:
                k = np.floor(r[:, 2] / anchor_dt)
                hit = np.flatnonzero(k[1:] > k[:-1])
                if hit.size:
                    i = hit[0]
                    t0, t1 = r[i, 2], r[i + 1, 2]
                    f = (k[i + 1] * anchor_dt - t0) / max(t1 - t0, 1e-12)
                    arc = arc - (arc[i] + (arc[i + 1] - arc[i]) * f)
            parts.append(np.column_stack((x[:-1], y[:-1], x[1:], y[1:],
                                          r[:-1, 3], r[1:, 3], arc[:-1], arc[1:])))
        if not parts:
            return np.zeros((0, _SEG_FLOATS), dtype='f4')
        return np.ascontiguousarray(np.concatenate(parts, axis=0), dtype='f4')

    # ---- zeichnen ---------------------------------------------------------

    def _draw_prediction_styled(self, sampled_runs, t_head, t_end):
        """Zeichnet die linie in der pixel-zeichenweise. False = nicht
        gezeichnet (klassisch, oder ein shader fehlt) -> alte linie."""
        style = self._pred_style()
        self.debug_info.pop('pred_pixel_error', None)
        if style == 'klassisch' or not self._pred_pixel_ready():
            self.debug_info['pred_pixel_marks'] = 0
            self.debug_info['pred_pixel_mark_dt'] = None
            return False
        runs = self._pred_runs_with_time(sampled_runs, t_head, t_end)
        if not runs:
            return True
        s = self._pred_settings(style)
        code = _STYLE_CODE[style]

        # Zeittakt der marken (und der anker des ditherns).
        pxps = self._pred_px_per_s(runs)
        spacing = {'perlen': 'spacing', 'lineal': 'spacing', 'schiene': 'spacing'}.get(style)
        mark_dt = None
        marks = np.zeros((0, 6))
        if spacing is not None:
            min_px = float(self._pred_px(s[spacing]))
            key = (style, min_px)
            current = self._pred_mark_dt if getattr(self, '_pred_mark_key', None) == key else None
            mark_dt = _pick_dt(pxps, min_px, current)
            self._pred_mark_dt, self._pred_mark_key = mark_dt, key
            if mark_dt is not None:
                marks = self._pred_crossings(runs, mark_dt)
                if len(marks) > _MAX_MARKS:
                    marks = marks[:0]
        anchor_dt = mark_dt or _pick_dt(pxps, _ANCHOR_SPACING_PX, None)
        segs = self._pred_segments(runs, anchor_dt)
        if len(segs) == 0:
            return True

        # Breiten in ganzen pixeln.
        if style == 'kontur':
            w, outline = self._pred_px(s['width']), float(self._pred_px(s['outline'], 0))
            aux, aux2 = float(s['outline_alpha']), 0.0
        elif style == 'perlen':
            w, outline, aux, aux2 = 1, 0.0, float(s['thread_alpha']), 0.0
        elif style == 'lineal':
            w, outline, aux, aux2 = self._pred_px(s['width']), 0.0, 0.0, 0.0
        elif style == 'halbton':
            w, outline = self._pred_px(s['width']), 0.0
            aux, aux2 = float(s['min_coverage']), float(max(1, int(s['steps'])))
        else:  # schiene
            w, outline = self._pred_px(s['width']), 1.0
            aux, aux2 = float(s['rail_alpha']), float(s['outline_alpha'])
        reach = 0.5 * w + outline + 1.5

        fade_steps = float(max(1, int(getattr(self, 'prediction_line_fade_steps', 4))))
        fade_min = float(getattr(self, 'prediction_line_fade_min', 0.42))

        # Mark-geometrie.
        mreach = 0.0
        mark_rows = None
        if len(marks):
            if style == 'perlen':
                size = self._pred_px(s['size'])
                every = max(1, int(s['major_every']))
                m1, m2, m3 = float(size), float(s['ring_alpha']), 0.0
                mreach = size + 3.0
            elif style == 'lineal':
                every = max(1, int(s['major_every']))
                m1 = float(self._pred_px(s['tick']))
                m2 = float(self._pred_px(s['major_tick']))
                m3 = 0.0
                mreach = max(m1, m2) + 2.0
            else:
                every = 1 << 30
                m1 = float(self._pred_px(s['arrow']))
                m2, m3 = 0.8, 0.0
                mreach = m1 + 3.0
            major = (np.mod(marks[:, 5], every) == 0).astype(np.float64)
            hh = float(self.height)
            mark_rows = np.ascontiguousarray(np.column_stack((
                marks[:, 0], hh - marks[:, 1], marks[:, 2], -marks[:, 3],
                marks[:, 4], major, np.zeros(len(marks)), np.zeros(len(marks)))),
                dtype='f4')

        # Kacheln, die linie und marken beruehren (GL, von unten).
        pad = max(reach, mreach) + 1.0
        tiles = self._pred_tiles(segs, mark_rows, pad)
        if len(tiles) == 0:
            return True
        W, H = int(self.width), int(self.height)

        ctx = self.ctx
        dest = ctx.fbo
        try:
            self._pred_ensure_buffer()
            self._pred_ensure_instances('seg', len(segs))
            if mark_rows is not None:
                self._pred_ensure_instances('mark', len(mark_rows))
            self._pred_ensure_tiles(len(tiles))
            self._pred_tile_vbo.write(tiles.tobytes())
            fbo = self._pred_buf_fbo
            fbo.use()
            fbo.viewport = (0, 0, W, H)

            ctx.blend_equation = moderngl.MAX
            ctx.blend_func = (moderngl.ONE, moderngl.ONE)
            prog = self._pred_seg_program
            prog['u_viewport'].value = (float(W), float(H))
            prog['u_reach'].value = float(reach)
            prog['u_style'].value = int(code)
            prog['u_w'].value = float(w)
            prog['u_outline'].value = float(outline)
            prog['u_aux'].value = (float(aux), float(aux2))
            prog['u_fade'].value = (fade_steps, fade_min)
            self._pred_seg_vbo.write(segs.tobytes())
            self._pred_seg_vao.render(moderngl.TRIANGLE_STRIP, vertices=4,
                                      instances=len(segs))
            if mark_rows is not None:
                mp = self._pred_mark_program
                mp['u_viewport'].value = (float(W), float(H))
                mp['u_reach'].value = float(mreach)
                mp['u_style'].value = int(code)
                mp['u_m'].value = (m1, m2, m3)
                mp['u_fade'].value = (fade_steps, fade_min)
                self._pred_mark_vbo.write(mark_rows.tobytes())
                self._pred_mark_vao.render(moderngl.TRIANGLE_STRIP, vertices=4,
                                           instances=len(mark_rows))
        except Exception as exc:
            self.debug_info['pred_pixel_error'] = f"{type(exc).__name__}: {exc}"
            self._pred_clear_all()
            ctx.blend_equation = moderngl.FUNC_ADD
            ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            dest.use()
            return False
        finally:
            ctx.blend_equation = moderngl.FUNC_ADD

        # Zusammenbau auf das ziel, kachelweise; danach dieselben kacheln im
        # puffer loeschen, damit er fuer das naechste frame leer ist.
        try:
            dest.use()
            line, mark = self._pred_colors()
            cp = self._pred_compose_program
            cp['u_viewport'].value = (float(W), float(H))
            cp['u_tile'].value = float(_TILE)
            cp['u_line'].value = tuple(float(c) for c in line)
            cp['u_rail'].value = tuple(float(c) * 0.62 for c in line)
            cp['u_mark'].value = tuple(float(c) for c in mark)
            cp['u_ground'].value = _GROUND
            self._pred_buf_tex.use(location=0)
            ctx.blend_func = (moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA)
            self._pred_compose_vao.render(moderngl.TRIANGLE_STRIP, vertices=4,
                                          instances=len(tiles))
            self._pred_buf_fbo.use()
            ctx.disable(moderngl.BLEND)
            clp = self._pred_clear_program
            clp['u_viewport'].value = (float(W), float(H))
            clp['u_tile'].value = float(_TILE)
            self._pred_clear_vao.render(moderngl.TRIANGLE_STRIP, vertices=4,
                                        instances=len(tiles))
        except Exception as exc:
            self.debug_info['pred_pixel_error'] = f"{type(exc).__name__}: {exc}"
            self._pred_clear_all()
        finally:
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            dest.use()
        self.debug_info['pred_pixel_marks'] = int(len(marks))
        self.debug_info['pred_pixel_mark_dt'] = mark_dt
        return True
