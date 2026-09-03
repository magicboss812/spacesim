"""Die Numba-fassungen der reinen zahlenschleifen im linien-zeichenweg.

Min-step-verdichtung, RDP-vereinfachung, fenster-clipping und verdichtung.
Wort-fuer-wort dieselbe arithmetik wie die Python-methoden in
`render/prediction.py` -- die bleiben als referenz und fallback erhalten;
ohne numba aendert sich exakt nichts ausser der geschwindigkeit.

`tests/prediction_projection_test.py` §1 zieht dieselbe linie durch den
batch- und den skalar-weg und verlangt 0.000e+00 px unterschied. Nach jeder
aenderung an einem dieser kerne ist das der test, der laufen muss.
"""
import math

import numpy as np


# Numba-fassungen der reinen zahlenschleifen im linien-zeichenweg
# (min-step-verdichtung und RDP-vereinfachung). Wort-fuer-wort dieselbe
# arithmetik wie die Python-methoden darunter -- die bleiben als referenz
# und fallback erhalten; ohne numba aendert sich exakt nichts.
try:
    from numba import njit as _njit

    @_njit(cache=True, nogil=True)
    def _compact_min_step_numba(xs, ys, min_step2):
        n = xs.shape[0]
        keep = np.empty(n, dtype=np.int64)
        keep[0] = 0
        m = 1
        lx = xs[0]
        ly = ys[0]
        for i in range(1, n):
            dx = xs[i] - lx
            dy = ys[i] - ly
            if dx * dx + dy * dy >= min_step2:
                keep[m] = i
                m += 1
                lx = xs[i]
                ly = ys[i]
        # Wie die Python-fassung: der letzte punkt wird angehaengt, wenn er
        # nicht ohnehin schon der zuletzt behaltene ist (koordinatenvergleich).
        if xs[keep[m - 1]] != xs[n - 1] or ys[keep[m - 1]] != ys[n - 1]:
            keep[m] = n - 1
            m += 1
        return keep[:m]

    @_njit(cache=True, nogil=True)
    def _rdp_keep_numba(xs, ys, tol2):
        n = xs.shape[0]
        keep = np.zeros(n, dtype=np.uint8)
        keep[0] = 1
        keep[n - 1] = 1
        stack = np.empty((2 * n + 8, 2), dtype=np.int64)
        stack[0, 0] = 0
        stack[0, 1] = n - 1
        top = 1
        while top > 0:
            top -= 1
            start = stack[top, 0]
            end = stack[top, 1]
            if end <= start + 1:
                continue

            ax = xs[start]
            ay = ys[start]
            bx = xs[end]
            by = ys[end]
            abx = bx - ax
            aby = by - ay
            ab2 = abx * abx + aby * aby

            max_d2 = -1.0
            index = -1
            for i in range(start + 1, end):
                px = xs[i]
                py = ys[i]
                if ab2 <= 1e-18:
                    dx = px - ax
                    dy = py - ay
                    d2 = dx * dx + dy * dy
                else:
                    apx = px - ax
                    apy = py - ay
                    t = (apx * abx + apy * aby) / ab2
                    if t < 0.0:
                        t = 0.0
                    elif t > 1.0:
                        t = 1.0
                    proj_x = ax + t * abx
                    proj_y = ay + t * aby
                    dx = px - proj_x
                    dy = py - proj_y
                    d2 = dx * dx + dy * dy
                if d2 > max_d2:
                    max_d2 = d2
                    index = i

            if max_d2 > tol2 and index != -1:
                keep[index] = 1
                stack[top, 0] = start
                stack[top, 1] = index
                top += 1
                stack[top, 0] = index
                stack[top, 1] = end
                top += 1
        return keep

    @_njit(cache=True, nogil=True)
    def _clip_runs_numba(xs, ys, left, top, right, bottom):
        """Liang-Barsky ueber die GANZE polylinie, laufweise zerlegt.

        Wort-fuer-wort dieselbe zustandsmaschine wie
        `Renderer._build_clipped_polyline_runs`: ein segment, das das
        rechteck nicht schneidet, beendet den laufenden lauf; eine luecke
        von mehr als 2 px zwischen dem ende des laufs und dem anfang des
        naechsten geklippten segments ebenfalls.

        Rueckgabe: (ox, oy, starts, counts). Lauf k sind die punkte
        ox[starts[k] : starts[k]+counts[k]].
        """
        n = xs.shape[0]
        # Je segment werden hoechstens zwei punkte geschrieben. Bei n < 2
        # laeuft die schleife nicht und alle puffer bleiben leer -- kein
        # sonderweg noetig (und damit ein einziger rueckgabetyp fuer numba).
        max_out = 2 * (n - 1) if n > 1 else 0
        ox = np.empty(max_out, dtype=np.float64)
        oy = np.empty(max_out, dtype=np.float64)
        starts = np.empty(n, dtype=np.int64)
        counts = np.empty(n, dtype=np.int64)

        n_runs = 0
        m = 0
        run_start = -1

        for i in range(n - 1):
            x0 = xs[i]
            y0 = ys[i]
            dx = xs[i + 1] - x0
            dy = ys[i + 1] - y0

            u1 = 0.0
            u2 = 1.0
            inside = True

            # links
            pi = -dx
            qi = x0 - left
            if pi == 0.0:
                if qi < 0.0:
                    inside = False
            else:
                t = qi / pi
                if pi < 0.0:
                    if t > u2:
                        inside = False
                    elif t > u1:
                        u1 = t
                elif t < u1:
                    inside = False
                elif t < u2:
                    u2 = t

            if inside:
                # rechts
                pi = dx
                qi = right - x0
                if pi == 0.0:
                    if qi < 0.0:
                        inside = False
                else:
                    t = qi / pi
                    if pi < 0.0:
                        if t > u2:
                            inside = False
                        elif t > u1:
                            u1 = t
                    elif t < u1:
                        inside = False
                    elif t < u2:
                        u2 = t

            if inside:
                # oben
                pi = -dy
                qi = y0 - top
                if pi == 0.0:
                    if qi < 0.0:
                        inside = False
                else:
                    t = qi / pi
                    if pi < 0.0:
                        if t > u2:
                            inside = False
                        elif t > u1:
                            u1 = t
                    elif t < u1:
                        inside = False
                    elif t < u2:
                        u2 = t

            if inside:
                # unten
                pi = dy
                qi = bottom - y0
                if pi == 0.0:
                    if qi < 0.0:
                        inside = False
                else:
                    t = qi / pi
                    if pi < 0.0:
                        if t > u2:
                            inside = False
                        elif t > u1:
                            u1 = t
                    elif t < u1:
                        inside = False
                    elif t < u2:
                        u2 = t

            if not inside:
                if run_start >= 0:
                    cnt = m - run_start
                    if cnt >= 2:
                        starts[n_runs] = run_start
                        counts[n_runs] = cnt
                        n_runs += 1
                    else:
                        m = run_start
                    run_start = -1
                continue

            cx0 = x0 + u1 * dx
            cy0 = y0 + u1 * dy
            cx1 = x0 + u2 * dx
            cy1 = y0 + u2 * dy

            if run_start < 0:
                run_start = m
                ox[m] = cx0
                oy[m] = cy0
                m += 1
                ox[m] = cx1
                oy[m] = cy1
                m += 1
                continue

            gx = cx0 - ox[m - 1]
            gy = cy0 - oy[m - 1]
            if math.sqrt(gx * gx + gy * gy) > 2.0:
                cnt = m - run_start
                if cnt >= 2:
                    starts[n_runs] = run_start
                    counts[n_runs] = cnt
                    n_runs += 1
                else:
                    m = run_start
                run_start = m
                ox[m] = cx0
                oy[m] = cy0
                m += 1
                ox[m] = cx1
                oy[m] = cy1
                m += 1
            else:
                ox[m] = cx1
                oy[m] = cy1
                m += 1

        if run_start >= 0:
            cnt = m - run_start
            if cnt >= 2:
                starts[n_runs] = run_start
                counts[n_runs] = cnt
                n_runs += 1
            else:
                m = run_start

        return (ox, oy, starts[:n_runs], counts[:n_runs])

    @_njit(cache=True, nogil=True)
    def _max_gap_refine_numba(keep_idx, xs, ys, max_seg):
        """Zu weit auseinanderliegende RDP-punkte wieder auffuellen.

        Dieselbe schleife wie die Python-fassung in
        `_runs_from_screen_points`, inklusive der bankier-rundung von
        Pythons `round()` -- ein um eins verschobener stuetzindex waere
        eine andere linie.
        """
        k = keep_idx.shape[0]
        # Die ausgabe ist streng monoton steigend und bleibt unter dem
        # letzten keep-index, kann also nie mehr als n punkte umfassen.
        out = np.empty(xs.shape[0] + k + 1, dtype=np.int64)
        out[0] = keep_idx[0]
        m = 1
        for i in range(1, k):
            start_idx = out[m - 1]
            end_idx = keep_idx[i]
            if end_idx <= start_idx:
                continue

            sdx = xs[end_idx] - xs[start_idx]
            sdy = ys[end_idx] - ys[start_idx]
            seg_len = math.sqrt(sdx * sdx + sdy * sdy)

            if seg_len > max_seg:
                steps = int(math.ceil(seg_len / max_seg))
                if steps < 2:
                    steps = 2
                span = end_idx - start_idx
                for step_i in range(1, steps):
                    v = span * (step_i / steps)
                    fl = math.floor(v)
                    frac = v - fl
                    if frac > 0.5:
                        cand = int(fl) + 1
                    elif frac < 0.5:
                        cand = int(fl)
                    else:
                        # Python rundet die haelfte zur GERADEN zahl.
                        fi = int(fl)
                        cand = fi if (fi % 2 == 0) else fi + 1
                    cand += start_idx
                    if cand <= out[m - 1]:
                        cand = out[m - 1] + 1
                    if cand >= end_idx:
                        break
                    out[m] = cand
                    m += 1

            if end_idx > out[m - 1]:
                out[m] = end_idx
                m += 1

        return out[:m]

    @_njit(cache=True, nogil=True)
    def _densify_numba(xs, ys, max_segment):
        """Segmente laenger als `max_segment` linear unterteilen.

        Zwei durchgaenge (zaehlen, fuellen) statt einer wachsenden liste --
        rechnerisch identisch zu `Renderer._densify_screen_run`.
        """
        n = xs.shape[0]
        total = 1 if n > 0 else 0
        for i in range(n - 1):
            dx = xs[i + 1] - xs[i]
            dy = ys[i + 1] - ys[i]
            seg_len = math.sqrt(dx * dx + dy * dy)
            if seg_len > max_segment:
                parts = int(math.ceil(seg_len / max_segment))
                if parts < 2:
                    parts = 2
                elif parts > 256:
                    parts = 256
                total += parts - 1
            total += 1

        dx_out = np.empty(total, dtype=np.float64)
        dy_out = np.empty(total, dtype=np.float64)
        m = 0
        if n > 0:
            dx_out[0] = xs[0]
            dy_out[0] = ys[0]
            m = 1
        for i in range(n - 1):
            x0 = xs[i]
            y0 = ys[i]
            dx = xs[i + 1] - x0
            dy = ys[i + 1] - y0
            seg_len = math.sqrt(dx * dx + dy * dy)
            if seg_len > max_segment:
                parts = int(math.ceil(seg_len / max_segment))
                if parts < 2:
                    parts = 2
                elif parts > 256:
                    parts = 256
                for p in range(1, parts):
                    t = p / parts
                    dx_out[m] = x0 + dx * t
                    dy_out[m] = y0 + dy * t
                    m += 1
            dx_out[m] = xs[i + 1]
            dy_out[m] = ys[i + 1]
            m += 1

        return (dx_out[:m], dy_out[:m])

    _LINE_KERNELS_OK = True
except Exception:
    # OHNE NUMBA MUESSEN DIE NAMEN TROTZDEM EXISTIEREN.
    #
    # Solange diese kerne mit dem Renderer in EINER datei lagen, genuegte das
    # flag: die aufrufstellen sind alle mit `if _LINE_KERNELS_OK:` bewacht, und
    # ein nie ausgewerteter name stoert nicht. Als eigenes modul werden sie
    # jedoch mit `from render.line_kernels import ...` geholt -- und ein
    # fehlender name laesst dann schon den IMPORT scheitern, also den ganzen
    # renderer, statt nur den schnellpfad. Deshalb hier platzhalter.
    _LINE_KERNELS_OK = False
    _compact_min_step_numba = None
    _rdp_keep_numba = None
    _clip_runs_numba = None
    _max_gap_refine_numba = None
    _densify_numba = None
