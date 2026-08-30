"""
OpenGL-Renderer für die Weltraumsimulation.
Verwendet pygame für Fensterverwaltung und HUD, moderngl (OpenGL) für Rendering.
"""

import pygame
from pygame.locals import *
import moderngl
import math
import os
import struct
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np

from reference_frames import IdentityReferenceFrame, apparent_orbital_directions
import background
import body_icon
import body_style
import orbit_lines
import ship_art
from background import BackgroundLayer

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
    _LINE_KERNELS_OK = False


class Renderer:
    def __init__(self, width, height, enable_fxaa=True, ctx=None):

        self.width = width
        self.height = height
        self.enable_fxaa = enable_fxaa

        # moderngl-context: hängt sich an den von pygame/SDL erstellten
        # GL-context. Aufrufer (test.py) können ihren bereits erstellten
        # wrapper übergeben, damit nicht zwei moderngl-contexte denselben
        # GL-state verwalten.
        self.ctx = ctx if ctx is not None else moderngl.create_context()

        # gpu-helpers: wiederverwendbare VBOs, programme und VAOs (erstellt in
        # _init_gpu_helpers). _quad_vbo (statisches einheits-quad) wird auch
        # vom FXAA-pfad genutzt und muss deshalb schon vor _init_fxaa
        # deklariert sein (lazy erstellt via _ensure_quad_vbo).
        self._poly_vbo = None
        self._poly_vbo_size = 0
        self._quad_vbo = None
        self._line_program = None
        self._line_vao = None
        self._background_program = None
        self._background_vao = None
        self._star_program = None
        self._star_vao = None
        self._star_vbo = None
        self._star_corner_vbo = None
        self._star_vbo_count = 0
        self._ortho_program = None
        self._ortho_vao = None
        # Zuletzt an die GL geschriebene uniform-/zustandswerte, siehe
        # _set_uniform / _set_line_width.
        self._line_viewport = None
        self._line_color = None
        self._ortho_viewport = None
        self._ortho_color = None
        self._texquad_viewport = None
        self._texquad_color = None
        self._background_viewport = None
        self._star_viewport = None
        self._gl_line_width = None
        self._body_program = None
        self._body_vao = None

        # --- prozedurale vektor-optik der koerper (D2) -------------------
        # Die zeichnung eines koerpers ist geometrie, keine textur: sie wird
        # einmal je (seed, farbe, muster) gebaut, liegt im einheitskreis und
        # wird pro frame nur skaliert. Beleuchtung ist ein uniform, deshalb
        # wandert der terminator mit der bahn, ohne dass etwas neu entsteht.
        self._body_surface_program = None
        self._body_line_program = None
        self._body_style_gpu = {}
        # Die positions-marke: EIN geteiltes quad fuer alle koerper, das
        # zellmuster steckt in vier uint32 als uniform. Deshalb kein puffer
        # je koerper und nichts, was pro frame belegt wuerde.
        self._body_icon_program = None
        self._body_icon_vao = None
        self._body_icon_cache = {}
        self._icon_viewport = None
        self._icon_tier_alpha = None
        self._icon_grid = None
        self._icon_edge = None
        self._icon_gap = None
        self._icon_rim = None
        self._icon_rim_dark = None
        self._icon_shade = None
        self._icon_halo = None
        self._icon_extent = None
        self._icon_radius = None
        self._icon_unit = None
        # Gebaut wird NEBENLAEUFIG. Der bau ist reine rechnung (numpy, keine
        # GL-aufrufe), nur das hochladen muss im hauptthread passieren --
        # gemessen der billige teil. Synchron gebaut kostete der erste frame
        # eines herangezoomten koerpers 18.5 ms; das ist genau der ruckler,
        # den man beim heranzoomen sieht, also dort, wo er auffaellt.
        self._body_style_jobs = {}
        self._body_style_executor = None
        # Gleichzeitige bauten. Einer reicht: mehr wuerden sich nur um die
        # GIL streiten und dem hauptthread dieselbe zeit abziehen.
        self._body_style_build_budget = 1
        self._light_source_body = None
        self._light_screen_xy = None
        self._texquad_program = None
        self._texquad_vao = None

        # FXAA framebuffer, textur, shader-programm und VAO
        self.fbo = None
        self.fbo_texture = None
        self.fxaa_program = None
        self._fxaa_vao = None

        # OpenGL initialisieren
        self._init_opengl()

        # FXAA initialisieren wenn aktiviert
        if self.enable_fxaa:
            self._init_fxaa()
        
        # UI-skalierung: die gesamte oberfläche wird in "design-einheiten" gegen
        # eine referenz-fensterhöhe angegeben und beim zeichnen mit ui_scale
        # multipliziert. Damit wächst das HUD auf großen/hochauflösenden
        # displays mit, statt bei 16 px stehenzubleiben.
        #
        # Die untergrenze ist bewusst 1.0: bei der default-fenstergröße
        # (kleiner als die referenz) bleibt die darstellung damit exakt so wie
        # bisher; skaliert wird ausschließlich nach oben.
        self.ui_scale_reference_height = 1000.0
        self.ui_scale_min = 1.0
        self.ui_scale_max = 3.0
        self.ui_scale_user = 1.0
        self.ui_scale = 1.0

        # Pygame Fonts für HUD. Die hier hinterlegten größen sind DESIGN-größen;
        # die tatsächlich erzeugten font-objekte tragen größe * ui_scale und
        # werden von _rebuild_fonts() bei jeder skalen-änderung neu erzeugt
        # (neu rastern statt hochskalieren -> text bleibt bei jeder auflösung scharf).
        pygame.font.init()
        self.hud_font_size_small = 16
        self.hud_font_size_medium = 20
        self.font_small = None
        self.font_medium = None
        self._recompute_ui_scale()
        self._rebuild_fonts()

        # Debug-Info
        self.debug_info = {
            'shader_error': None,
            'bodies_rendered': 0,
            'bodies_culled': 0,
            'bodies_as_icon': 0,
            'prediction_points_in': 0,
            'prediction_points_drawn': 0,
            'prediction_detail_target_m': None,
            'prediction_detail_achieved_m': None,
            'prediction_detail_added': 0,
        }
        self.render_benchmark_debug = False
        self.render_benchmark_every_n_frames = 60
        self._render_benchmark_frame = 0
        self._last_prediction_render_stats = {}
        # per-phase timings of the most recent render() call (frame_ms,
        # bodies_ms, swap_or_present_ms, ...). Read by the per-frame TIMING
        # line in test.py to split render calc vs. present cost.
        self.last_frame_timings = {}

        # optionales predictor-debug: wenn True druckt kleine beispiele der predictor-
        # punkte (bildschirm und rekonstruierte welt-koords) in die konsole.
        self.debug_predictor = False

        # principia-ähnliche visuelle sampling-kontrollen: linien-strip-rendering behalten,
        # aber punktdichte an bildschirm-krümmung/-fehler anpassen.
        self.prediction_sampling_tolerance_px = 1.5
        self.prediction_sampling_min_step_px = 0.35
        self.prediction_sampling_max_points = 1000
        # sehr feine bildschirm-toleranz beim reingezoomt erlauben.
        # kleinere werte ermöglichen mehr detail bei extremen zoom-stufen.
        self.prediction_sampling_min_tolerance_px = 0.005
        self.prediction_sampling_max_tolerance_px = 0.25
        self.prediction_sampling_max_segment_px = 4.0
        self.prediction_sampling_reference_scale = 1e-6
        self.prediction_visibility_margin_px = 128.0
        self.prediction_bypass_fxaa = True
        self.prediction_render_max_raw_scan = 3000
        self.prediction_render_max_draw_points = 4000
        self.prediction_render_max_world_length = None
        self.prediction_render_max_screen_length_px = None

        # ---- aufloesungsgetriebene verfeinerung der vorhersagelinie ----
        #
        # Die punkteliste ist seit den geschwindigkeits-spalten (predictor.
        # POINT_COLUMNS) eine stueckweise KUBISCHE kurve, keine folge von
        # positionen. Zwischen zwei stuetzstellen wird deshalb zur zeichenzeit
        # HERMITE ausgewertet statt linear verbunden -- und zwar nur so fein,
        # wie es der bildschirm hergibt, und nur dort, wo etwas zu sehen ist.
        #
        # Warum das ueberhaupt noetig ist: der kernel setzt punkte in festem
        # weltabstand (1000 km im auslieferungszustand). Eine sehne dieser
        # laenge schneidet eine erdnahe bahn um c^2/8R = 17.8 km ab. Kubisch
        # interpoliert sind es 7.6 m -- derselbe punktabstand, 2350-fach
        # kleinerer fehler, ohne einen einzigen zusaetzlichen
        # integrationsschritt.
        self.prediction_hermite_enabled = True
        # Ziel-abweichung in geraete-pixeln. `test.py` setzt
        # SDL_WINDOWS_DPI_AWARENESS=permonitorv2 vor dem display-init, also
        # sind self.width/height ECHTE geraetepixel -- ein pixel-budget ist
        # damit schon ein DPI-budget, ohne umrechnung.
        self.prediction_detail_scale = 1.0
        self.prediction_hermite_max_subdiv = 64
        # Sprossen der toleranz-leiter in metern, aufsteigend. Gewaehlt wird
        # die groesste sprosse, die noch unter dem bildschirm-wunsch liegt.
        #
        # Die quantisierung ist NICHT kosmetik: ohne sie aendert sich das ziel
        # bei jeder zoom-stufe und die verfeinerung wird jeden frame neu
        # gerechnet -- derselbe fehler, den der predictor bei
        # `snapshot_view_rel_tol` schon einmal hatte (37 neubauten je
        # zoom-geste statt 1).
        self.prediction_error_ladder_m = [0.001, 0.01, 1.0, 100.0, 1000.0]
        # apoapsis/periapsis-marker auf der prädiktionslinie (vom predictor
        # geliefert, hier nur gezeichnet).
        # Die alte debug-textwand. Standardmaessig aus -- das spieler-HUD
        # (spacesim/ui/hud/) traegt ihre werte seit Phase 4.
        self.show_debug_hud = False
        self.show_apsis_markers = True
        self.apsis_marker_radius_px = 5.0
        # Die marker blenden aus, wenn die bahn AM SCHIRM klein wird (nicht
        # nach zoom-schwelle): unter `fade_min_px` apsis-radius unsichtbar,
        # ab `fade_full_px` voll -- so ueberlagern sie bei weit-sicht nicht
        # die schiffs- und planeten-marken.
        self.apsis_marker_fade_min_px = 12.0
        self.apsis_marker_fade_full_px = 46.0
        self._prediction_line_cache_key_value = None
        self._prediction_line_cache_points = None
        self._prediction_line_cache_stats = {}
        self._prediction_frame_transform_debug_key = None
        self._current_body_index_by_id = {}
        self.current_reference_body = None

        # Körper-icon-schwelle (bildschirm-pixel). Sobald der ECHTE bildschirm-
        # radius eines (nicht-schiff-)körpers unter diesen wert fällt, wird der
        # volle körper (disc + glow + atmosphäre) de-rendert und stattdessen ein
        # positions-icon konstanter bildschirmgröße gezeichnet. Dieser eine wert
        # ist zugleich swap-schwelle UND icon-radius -> der körper schrumpft exakt
        # bis zu dieser größe und das icon übernimmt nahtlos bei identischer größe
        # (kein leerer frame, keine doppelzeichnung). Beim weiteren herauszoomen
        # bleibt das icon konstant groß (skaliert nicht mehr mit der zoom-stufe).
        # 8.0 statt der frueheren 4.0: die marke traegt ein zellmuster, und
        # bei 4 px waere eine zelle rund 1.1 px breit -- das ist kein muster
        # mehr, sondern Matsch. Bei 8 px sind es 3.2 px je zelle.
        # Die MINDESTgroesse -- zugleich die schwelle, unter der ein koerper
        # komplett gegen die marke getauscht wird (siehe body_icon_max_radius_px
        # und body_icon_size_influence fuer die obere seite der skalierung).
        self.body_icon_min_radius_px = 8.0

        # --- die positions-marke (body_icon.py) ---------------------------
        # `"pixel"` = das gesaete zellmuster, `"disc"` = die alte flache
        # scheibe. Die variante waehlt zwischen den beiden entwuerfen.
        self.body_icon_style = "pixel"
        self.body_icon_variant = body_icon.DEFAULT_VARIANT
        # Globaler seed-versatz: derselbe koerper bekommt bei jedem wert eine
        # eigene marke, ohne `style_seed` in solar_system.json anzufassen --
        # der schnelle weg, eine ganze neue serie durchzuprobieren.
        self.body_icon_seed_offset = 0
        # Der detailgrad. Groesser = mehr zellen, NICHT groessere marke.
        self.body_icon_grid = body_icon.DEFAULT_GRID
        # Bis hierher wird die marke ueber den echten koerper geblendet.
        # Ohne dieses band poppt der tausch: eine pixelmarke sieht nun einmal
        # anders aus als eine schattierte scheibe mit limbus, auch bei genau
        # gleichem radius.
        # Die HOECHSTgroesse, bis zu der die marke nach dem PHYSISCHEN
        # koerper-radius wachsen darf (siehe body_icon_size_influence).
        self.body_icon_max_radius_px = 48.0
        # Wie stark der physische koerper-radius die marken-groesse skaliert.
        # 0 = jede marke bleibt bei body_icon_min_radius_px (heutiges
        # verhalten), 1 = die marke folgt voll dem log-skalierten radius,
        # geklemmt auf [min, max]. Dazwischen linear gemischt.
        self.body_icon_size_influence = 0.0
        # Spanne der PHYSISCHEN koerper-radien im geladenen system (m), fuer
        # die log-skalierung -- `_update_icon_radius_range` setzt sie einmal
        # je frame aus der echten koerperliste. Der platzhalter hier greift
        # nur, solange noch kein frame gezeichnet wurde (z.b. in tests, die
        # `_body_icon_draw_radius_px` direkt aufrufen).
        self._icon_radius_range_m = (1.0, 1.0)
        # Das ueberblend-band endet bei body_icon_min_radius_px * diesem
        # FAKTOR (nicht bei einem absoluten pixelwert): eine absolute grenze
        # verlor zweimal den anschluss, als der radius von hand verstellt
        # wurde (min=32 mit dem alten fade=13 stand verkehrt herum; min=16
        # brauchte manuell nachgerechnete 25.6). Ein faktor > 1 kann das nicht
        # mehr, weil er sich am jeweils aktuellen minimum bemisst.
        self.body_icon_fade_factor = 1.6
        self.body_icon_halo_alpha = 0.30
        # Breite der umriss-glaettung in pixeln. 0 = harte kante (und damit
        # pixelweise bewegung), 1 = ein pixel deckungsrampe.
        self.body_icon_edge_px = 1.0
        # Anteil einer zelle, der als SPALT frei bleibt. Er macht aus einer
        # flaeche gleichfarbiger nachbarn wieder ein sichtbares raster --
        # dieselbe rolle wie `1 - 0.18*pixel_round` beim hintergrund-gitter.
        self.body_icon_cell_gap = 0.0
        # Streuung der Helligkeit je Zelle. Drei Stufen allein geben zu wenig
        # Tiefe -- gleich eingestufte Nachbarn verschmelzen sonst zu einer
        # Flaeche. Der Wert haengt nur an (Zelle, Seed) und flimmert deshalb
        # nicht, wenn sich die Marke bewegt.
        self.body_icon_shade_jitter = 0.30
        # Der UMRISS jeder Zelle: welcher Anteil ihrer Breite nachdunkelt und
        # wie stark. Das ist der "gemalte" Rand, der aus der Marke ein Raster
        # aus einzelnen Feldern macht -- in beiden Achsen gleich.
        self.body_icon_cell_rim = 1.0
        self.body_icon_cell_rim_dark = 0.42

        # --- wann ein koerper seinen namen zeigt --------------------------
        # `"selected"` (voreinstellung): nur der angewaehlte koerper wird
        # angeschrieben -- dafuer IMMER, auch wenn er weit herausgezoomt nur
        # noch als icon gezeichnet wird. `"zoom"` ist das alte verhalten
        # (jeder koerper ab `body_label_min_radius_px` bildschirmradius),
        # `"both"` beides zusammen.
        self.body_label_mode = "selected"
        self.body_label_min_radius_px = 5.0

        # --- schiffs-grafik (ship_art.py) ---------------------------------
        # Die vektor-zeichnung aus dem design-mockup. Sie wird wie der alte
        # pfeil in FESTEN bildschirm-pixeln gezeichnet: das schiff behaelt
        # seine groesse ueber jede zoomstufe hinweg, es ist bewusst KEINE
        # welt-geometrie (bei realistischem massstab waere es bei jeder
        # spielbaren zoomstufe kleiner als ein pixel).
        self.ship_sprite_enabled = True
        # Laenge (nase bis duesen-lippe) in DESIGN-einheiten; ui_px() rechnet
        # sie auf die aktuelle aufloesung um. `ship_render_scale` ist der
        # spieler-regler daneben (dev-UI, reiter "Ship").
        self.ship_length_px = 78.0
        self.ship_render_scale = 1.0
        self.ship_accent_color = ship_art.DEFAULT_ACCENT
        # Grundhelligkeit der abgasfahne ohne schub; unter schub faehrt sie
        # auf 1.0 hoch (siehe _draw_ship_sprite).
        self.ship_plume_idle = 0.22
        self._ship_geometry_cache = None
        self._ship_plume_level = 0.0
        # --- zoom-abhaengige verkleinerung des schiffs -------------------
        # Das schiff bleibt in bildschirm-pixeln gezeichnet (siehe oben),
        # aber NICHT ueber jede zoomstufe gleich gross: weit herausgezoomt --
        # wenn das ganze system ins bild passt und der koerper darunter
        # laengst nur noch ein icon ist -- ueberdeckt eine 78-px-silhouette
        # die halbe bahn. Ab `ship_zoom_shrink_start_scale` (px je meter)
        # faehrt der massstab deshalb nach unten, bis er bei
        # `ship_zoom_shrink_end_scale` auf `ship_zoom_shrink_min` steht.
        #
        # Die ueberblendung ist bewusst KEINE stufe und auch nicht linear in
        # der zoomstufe: sie laeuft als smoothstep im LOG-raum der skala
        # (zoom ist multiplikativ, genau wie camera._ease_scale es rechnet),
        # damit die groesse beim durchzoomen weich wandert statt an einer
        # schwelle umzuspringen.
        self.ship_zoom_shrink_enabled = True
        self.ship_zoom_shrink_start_scale = 1e-6   # darueber: volle groesse
        self.ship_zoom_shrink_end_scale = 1e-9     # darunter: minimalgroesse
        self.ship_zoom_shrink_min = 0.55
        # Je frame in render() aus camera.scale nachgezogen, damit alle
        # zeichenwege (grafik, pfeil-fallback, label-abstand) denselben
        # faktor sehen -- sie bekommen die kamera nicht alle uebergeben.
        self._ship_zoom_factor = 1.0

        # --- auswahl-markierung (vier pfeile um den angeklickten koerper) --
        # Alle groessen sind DESIGN-einheiten und laufen durch ui_px(), also
        # bildschirm-groessen wie beim schiffspfeil und bei den linienbreiten
        # der koerper-optik: die markierung ist bei jeder zoomstufe und jeder
        # aufloesung gleich gross, sie ist keine welt-geometrie.
        self.selection_marker_enabled = True
        self.selection_marker_color = (0.36, 0.86, 0.98, 0.92)
        self.selection_arrow_length_px = 13.0   # spitze -> basis
        self.selection_arrow_width_px = 11.0    # basisbreite
        self.selection_gap_px = 7.0             # spitze <-> koerperrand
        self.selection_min_radius_px = 9.0      # bei icon-grossen koerpern
        self.selection_max_radius_px = 260.0    # bildfuellender koerper
        self.selection_spin_deg_per_s = 22.0
        self.selection_pulse_period_s = 1.6
        self.selection_pulse_amount = 0.07
        # Zusaetzlicher greifrand beim anklicken (design-einheiten). Ohne ihn
        # waere ein 4-px-icon ein 4-px-ziel.
        self.selection_pick_margin_px = 10.0
        # Laufende phasen der markierung. Sie werden mit dem ECHTEN frame-delta
        # fortgeschrieben (nicht je frame um einen festen betrag), damit die
        # bewegung bei 30 wie bei 240 fps gleich schnell ist -- dieselbe regel
        # wie bei jeder anderen bewegung im projekt.
        self._selection_spin_phase = 0.0
        self._selection_pulse_phase = 0.0
        self.selected_body = None

        # --- schwellen und regler der koerper-optik ----------------------
        # Unterhalb von `body_vector_min_radius_px` sieht man von facetten
        # ohnehin nichts, also wird gar nichts gebaut und der koerper bleibt
        # die alte flache scheibe. Dazwischen blendet `u_fade` linear ein --
        # ein harter schnitt bei einer zoomstufe waere als aufblitzen sichtbar.
        self.body_vector_style = True
        self.body_vector_min_radius_px = 11.0
        self.body_vector_full_radius_px = 26.0
        # Detailleiter. Die stufe wird NICHT fest gewaehlt, sondern so, dass
        # eine facette immer ungefaehr `body_vector_facet_px` pixel breit ist.
        #
        # Das ist nicht bloss sparsamkeit. Fest auf 'fine' sieht ein koerper
        # mit 40 px radius aus wie ein golfball: 26 facetten ueber den
        # durchmesser sind dort 3 px breit und werden zu grauem rauschen.
        # Fest auf 'coarse' fehlt beim heranzoomen genau die zeichnung, um
        # derentwillen das ganze gebaut wurde. Gemessen kostet der bau
        # 2.0 / 4.0 / 13.0 ms und belegt 0.12 / 0.32 / 1.13 MB je koerper --
        # einmalig, danach ist es reine zeichenarbeit.
        #
        # Der uebergang wird UEBERBLENDET (`body_vector_detail_blend`), sonst
        # springt das muster mitten in einer zoom-geste um.
        self.body_vector_detail = None  # 'coarse'/'medium'/'fine' erzwingt eine stufe
        self.body_vector_facet_px = 14.0
        self.body_vector_detail_blend = 0.35
        self.body_vector_coverage = 0.5
        self.body_vector_shape_density = body_style.DEFAULT_SHAPE_DENSITY
        # Beleuchtung: die richtung kommt vom stern, liegt in der bahnebene
        # (z = 0) und ergibt damit die echte phase. `body_ambient` ist die
        # resthelligkeit der nachtseite -- ohne sie verschwindet die haelfte
        # des koerpers restlos.
        self.body_light_enabled = True
        self.body_ambient = 0.16
        self.body_light_exponent = 1.45
        # Anteil des lichts, der ZUM betrachter zeigt.
        #
        # 0.0 waere die physikalisch exakte phase fuer eine draufsicht auf die
        # bahnebene -- dann steht der subsolare punkt aber IMMER auf dem rand,
        # die scheibenmitte liegt genau auf dem terminator, und jeder planet
        # ist auf ewig halb. Gemessen bleibt davon wenig uebrig: die hellste
        # stelle ist die staerkste verkuerzung, der koerper liest sich als
        # dunkler fleck. 0.55 ist der wert des mockups und kippt das bild in
        # eine dreiviertel-beleuchtung -- eine seite klar heller als die
        # andere, was der eigentliche zweck ist.
        self.body_light_tilt = 0.55
        # Grundglimmen jedes koerpers in seiner eigenen farbe. Sterne bekommen
        # ueber `light_intensity` zusaetzlich einen groesseren halo.
        self.body_glow_alpha = 0.16

        # Bildschirm-bounding-box kleiner als dieser wert (px) => referenz-spur
        # wird nicht gezeichnet (sub-pixel, ohnehin unsichtbar).
        self.reference_traj_min_screen_px = 2.0

        # frame-status (principia-ähnlich): physik bleibt absolut, rendering
        # wendet den aktuell ausgewählten plotting-frame plus optionales target-
        # overlay-frame an.
        self._plotting_frame = IdentityReferenceFrame()
        self._plotting_frame_label = "Barycentric"
        self._target_frame = None
        self._target_frame_label = None
        self._frame_time_s = 0.0
        # debugging: aktivieren um periodisch aktives frame und ausgewählte
        # körper welt/frame-koordinaten zur inspektion zu drucken.
        self.debug_frame = False
        self._frame_debug_counter = 0
        self._frame_debug_period = 30

        # reference-frame trajectorien-spuren (historie im frame-raum).
        # diese ersetzen statische scripted-orbit-ellipsen und zeigen relative
        # epizykel-bewegung für alle körper im aktiven frame.
        self.reference_trajectories_enabled = True
        self.reference_trajectories_max_points = 2400
        self.reference_trajectories_sample_step_s = 1.0
        self._reference_traj_last_sample_time = None
        self._reference_traj_points = {}

        # Hintergrund-ebene (background.py): sternenfeld + dreiecksgitter.
        # Reiner zustand, kein GL -- die GL-seite haengt an
        # _init_background_pipeline / _draw_background. Alle schalter darin
        # sind zugleich die schluessel des `background`-abschnitts der
        # config und die regler des ImGui-panels.
        self.background = BackgroundLayer()

        # Bahnlinien der koerper (orbit_lines.py). Die deckkraft kommt aus
        # der dichtesten annaeherung der praediktor-linie an die ZUKUENFTIGE
        # position des koerpers, gemessen in vielfachen seiner
        # einflusssphaere -- so heisst "nah" beim Mond dasselbe wie bei
        # Jupiter.
        self.orbit_lines_enabled = True
        self.orbit_line_tolerance_px = 0.3
        self.orbit_line_min_screen_px = 3.0
        self.orbit_line_track_samples = 192
        self.orbit_line_soi_full = 1.0
        self.orbit_line_soi_fade = 3.0
        self.orbit_line_reveal_full = 10.0
        self.orbit_line_reveal_fade = 30.0
        self.orbit_line_alpha_max = 0.85
        self.orbit_line_alpha_floor = 0.12
        self.orbit_line_alpha_floor_focus = 0.35
        self.orbit_line_fade_rate = 6.0
        self.orbit_line_width = 1.6
        self.orbit_line_knot_angle = 0.05
        self.orbit_line_end_caps = True
        self.orbit_line_end_cap_px = 4.5
        # Faint volllinie: EIN ganzer umlauf des koerpers, hinter der hellen
        # enthuellten spur, mit regelbarer (niedriger) deckkraft. Eigene,
        # groebere knoten-tabelle -- das fenster ist eine ganze periode.
        self.orbit_line_full_orbit_enabled = True
        self.orbit_line_full_alpha_mult = 0.30
        self.orbit_line_full_knot_angle = 0.12
        self.orbit_line_full_samples = 256
        self.orbit_line_full_max_span_s = 7.5e7
        self._orbit_line_set = None
        # Knotentabellen der faint volllinien, eine je (rahmen, fenster),
        # ueber die frames gehalten. Siehe _draw_orbit_lines.
        self._full_orbit_tables = {}
        self._shader_dir = os.path.join(os.path.dirname(__file__), 'shaders')

        self._label_texture_cache = {}
        # obergrenze für gecachte label-texturen: ständig wechselnde texte
        # (z. B. das schiffs-speed-label, das sich fast jeden frame ändert)
        # würden sonst unbegrenzt GL-texturen anhäufen (vram-leak).
        self._label_texture_cache_max = 256
        # Verdraengte texturen nach groesse, siehe _acquire_label_texture.
        self._label_texture_pool = {}
        self._label_texture_pool_count = 0
        # pro-frame-memo der kamera-position im aktiven frame: dieselbe
        # transformation wird sonst von _draw_body (pro körper!), trails und
        # prediction mehrfach pro frame berechnet.
        self._camera_frame_xy_key = None
        self._camera_frame_xy_value = (0.0, 0.0)
        # cache der pro-zeile gerenderten HUD-surfaces: ändert sich nur eine
        # zeile (z. B. kamera-position), müssen die übrigen nicht erneut durch
        # font.render laufen.
        self._hud_line_surface_cache = {}
        self._hud_texture = None
        self._hud_texture_size = (0, 0)
        # Körper-beschriftungen, während des FBO-durchgangs gesammelt und erst
        # nach dem FXAA-resolve gezeichnet (siehe _draw_body).
        self._deferred_labels = []
        # HUD-memoization: solange die formatierten textzeilen identisch sind,
        # bleibt die persistente HUD-textur gültig und muss weder neu gerastert
        # (font.render/Surface/tostring) noch hochgeladen werden.
        self._hud_cache_key = None
        # GPU-helpers initialisieren (VBOs, programme, VAOs). Kein blanket-
        # try/except mehr: ohne diese pipelines gibt es keinen fixed-function-
        # fallback, ein fehler hier soll sofort sichtbar sein. Einzelne
        # pipelines degradieren weiterhin kontrolliert (programm = None).
        self._init_gpu_helpers()
    
    def _init_opengl(self):
        """Initialisiert OpenGL-Einstellungen (moderngl-state)."""
        self.ctx.viewport = (0, 0, self.width, self.height)

        # Blending aktivieren; depth test wird nie aktiviert (2D)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        # Hintergrundfarbe (dunkelblau)
        self._clear_color = (0.0, 0.0, 0.05, 1.0)

        # VSync kommt vom fenster-swap: pygame.display.set_mode(..., vsync=1)
        # bzw. SDL_VIDEO_VSYNC in test.py. Der alte wgl/glX-hack entfällt.

    def _create_fxaa_targets(self):
        """Erstellt FBO-textur und framebuffer in aktueller fenstergröße."""
        self.fbo_texture = self.ctx.texture((self.width, self.height), 4)
        self.fbo_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.fbo_texture.repeat_x = False  # CLAMP_TO_EDGE
        self.fbo_texture.repeat_y = False
        self.fbo = self.ctx.framebuffer(color_attachments=[self.fbo_texture])

    def _release_fxaa_targets(self):
        for name in ('fbo', 'fbo_texture'):
            obj = getattr(self, name, None)
            if obj is not None:
                try:
                    obj.release()
                except Exception:
                    pass
            setattr(self, name, None)

    def _init_fxaa(self):
        """Initialisiert FXAA Framebuffer und Shader."""
        try:
            self._create_fxaa_targets()

            # FXAA Shader laden
            self._load_fxaa_shaders()

            print("FXAA initialized successfully")
        except Exception as e:
            print(f"FXAA initialization failed: {e}")
            self._release_fxaa_targets()
            self.enable_fxaa = False
    
    def _load_fxaa_shaders(self):
        fxaa_vertex_source = """
        #version 330
        in vec2 a_pos;
        out vec2 v_texcoord;
        void main() {
            v_texcoord = a_pos * 0.5 + 0.5;
            gl_Position = vec4(a_pos, 0.0, 1.0);
        }
        """

        fxaa_fragment_source = """
        #version 330
        uniform sampler2D u_texture;
        uniform vec2 u_resolution;
        in vec2 v_texcoord;
        out vec4 fragColor;

        float luminance(vec3 c) {
            return dot(c, vec3(0.299, 0.587, 0.114));
        }

        void main() {
            vec2 texel_size = 1.0 / u_resolution;
            vec2 uv = v_texcoord;

            vec3 center = texture(u_texture, uv).rgb;
            float center_luma = luminance(center);

            vec3 nw = texture(u_texture, uv + vec2(-1.0, -1.0) * texel_size).rgb;
            vec3 ne = texture(u_texture, uv + vec2(1.0, -1.0) * texel_size).rgb;
            vec3 sw = texture(u_texture, uv + vec2(-1.0, 1.0) * texel_size).rgb;
            vec3 se = texture(u_texture, uv + vec2(1.0, 1.0) * texel_size).rgb;

            float luma_nw = luminance(nw);
            float luma_ne = luminance(ne);
            float luma_sw = luminance(sw);
            float luma_se = luminance(se);

            float luma_min = min(center_luma, min(min(luma_nw, luma_ne), min(luma_sw, luma_se)));
            float luma_max = max(center_luma, max(max(luma_nw, luma_ne), max(luma_sw, luma_se)));
            float luma_range = luma_max - luma_min;

            if (luma_range < 0.0312) {
                fragColor = vec4(center, 1.0);
                return;
            }

            float gradient_nw_se = abs(luma_nw - luma_se);
            float gradient_ne_sw = abs(luma_ne - luma_sw);
            float contrast = max(gradient_nw_se, gradient_ne_sw);

            if (contrast < 0.0625) {
                fragColor = vec4(center, 1.0);
                return;
            }

            vec2 dir;
            dir.x = -((luma_nw + luma_ne) - (luma_sw + luma_se));
            dir.y = ((luma_nw + luma_sw) - (luma_ne + luma_se));

            float dir_reduce = max((luma_nw + luma_ne + luma_sw + luma_se) * 0.25, 0.125);
            float rcp_dir_min = 1.0 / (min(abs(dir.x), abs(dir.y)) + dir_reduce);

            dir = min(vec2(8.0), max(vec2(-8.0), dir * rcp_dir_min)) * texel_size;

            vec3 result_a = 0.5 * (
                texture(u_texture, uv + dir * (1.0/3.0 - 0.5)).rgb +
                texture(u_texture, uv + dir * (2.0/3.0 - 0.5)).rgb
            );
            vec3 result_b = result_a * 0.5 + 0.25 * (
                texture(u_texture, uv + dir * -0.5).rgb +
                texture(u_texture, uv + dir * 0.5).rgb
            );

            float luma_b = luminance(result_b);

            if (luma_b < luma_min || luma_b > luma_max) {
                fragColor = vec4(result_a, 1.0);
            } else {
                fragColor = vec4(result_b, 1.0);
            }
        }
        """

        # moderngl kompiliert und linkt in einem schritt; compile-/link-fehler
        # werfen und werden vom aufrufer (_init_fxaa) behandelt.
        self.fxaa_program = self.ctx.program(
            vertex_shader=fxaa_vertex_source,
            fragment_shader=fxaa_fragment_source,
        )

        # Uniforms einmalig setzen (textur-unit 0; auflösung bei resize
        # aktualisiert) statt pro frame in _apply_fxaa.
        self.fxaa_program['u_texture'].value = 0
        self.fxaa_program['u_resolution'].value = (float(self.width), float(self.height))

        # Vollbild-quad (TRIANGLE_STRIP über das geteilte einheits-quad)
        self._fxaa_vao = self.ctx.vertex_array(
            self.fxaa_program, [(self._ensure_quad_vbo(), '2f', 'a_pos')]
        )

        print("FXAA Shader loaded successfully")

    def _apply_fxaa(self):
        """Wendet FXAA Post-Processing an.

        Erwartet, dass der ziel-framebuffer (screen) bereits gebunden ist.
        Das vollbild-quad überschreibt jeden pixel, daher ohne blending.
        """
        if not self.enable_fxaa or self.fbo_texture is None or self._fxaa_vao is None:
            return

        self.ctx.disable(moderngl.BLEND)
        self.fbo_texture.use(location=0)
        self._fxaa_vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.enable(moderngl.BLEND)

    def set_plotting_frame(self, frame, label=None):
        self._plotting_frame = frame if frame is not None else IdentityReferenceFrame()
        if label is not None:
            self._plotting_frame_label = str(label)
        else:
            self._plotting_frame_label = getattr(self._plotting_frame, 'label', 'Barycentric')
        self._reset_reference_trajectories()

    def set_target_frame(self, frame, label=None):
        self._target_frame = frame
        if frame is None:
            self._target_frame_label = None
            self._reset_reference_trajectories()
            return
        if label is not None:
            self._target_frame_label = str(label)
        else:
            self._target_frame_label = getattr(frame, 'label', 'Target overlay')
        self._reset_reference_trajectories()

    def clear_target_frame(self):
        self._target_frame = None
        self._target_frame_label = None
        self._reset_reference_trajectories()

    def set_frame_time(self, time_s):
        try:
            self._frame_time_s = float(time_s)
        except Exception:
            self._frame_time_s = 0.0

        for frame in (self._plotting_frame, self._target_frame):
            if frame is None:
                continue
            try:
                frame.set_epoch_time(self._frame_time_s)
            except Exception:
                pass

    def _active_frame(self):
        return self._target_frame if self._target_frame is not None else self._plotting_frame

    def _frame_transform_xy(self, x, y):
        frame = self._active_frame()
        try:
            return frame.to_this_frame_xy(self._frame_time_s, float(x), float(y))
        except Exception:
            return float(x), float(y)

    def _frame_camera_xy(self, camera):
        # memoisiert: ergebnis hängt nur von aktivem frame, frame-zeit und
        # kamera-position ab — alles konstant innerhalb eines render-frames.
        cam_x = float(camera.position.x)
        cam_y = float(camera.position.y)
        key = (id(self._active_frame()), self._frame_time_s, cam_x, cam_y)
        if key == self._camera_frame_xy_key:
            return self._camera_frame_xy_value
        value = self._frame_transform_xy(cam_x, cam_y)
        self._camera_frame_xy_key = key
        self._camera_frame_xy_value = value
        return value

    def _world_to_screen_xy(self, world_x, world_y, camera, camera_frame_xy=None):
        if camera_frame_xy is None:
            camera_frame_xy = self._frame_camera_xy(camera)
        frame_x, frame_y = self._frame_transform_xy(world_x, world_y)
        scale = float(camera.scale)
        sx = self.width * 0.5 + (frame_x - camera_frame_xy[0]) * scale
        sy = self.height * 0.5 - (frame_y - camera_frame_xy[1]) * scale
        return sx, sy

    def _world_to_screen_xy_at_time(self, world_x, world_y, camera, time_s, camera_frame_xy=None):
        """Konvertiert einen Welt-Punkt zu einer bestimmten Sim-Zeit in Bildschirmkoordinaten.

        Diese nutzt die zeitabhängige Transformation des aktiven Frames, sodass
        Prädiktor-Punkte (die pro Sample Sim-Zeiten enthalten) korrekt in einen
        sich bewegenden/rotierenden Plot-Frame projiziert werden.
        """
        frame = self._active_frame()
        try:
            frame_x, frame_y = frame.to_this_frame_xy(float(time_s), float(world_x), float(world_y))
        except Exception:
            # Fallback: auf aktuelle Frame-Transformation zurückfallen
            frame_x, frame_y = self._frame_transform_xy(world_x, world_y)

        # Keep the camera origin in the current render frame. Prediction samples
        # are time-tagged future world points; transforming the camera at the
        # sample time would cancel a moving reference-frame origin back out.
        if camera_frame_xy is None:
            camera_frame_xy = self._frame_camera_xy(camera)

        scale = float(camera.scale)
        sx = self.width * 0.5 + (frame_x - camera_frame_xy[0]) * scale
        sy = self.height * 0.5 - (frame_y - camera_frame_xy[1]) * scale
        return sx, sy

    def _prediction_frame_transform_mode(self):
        frame = self._active_frame()
        name = frame.__class__.__name__
        label = str(getattr(frame, 'label', '') or '')
        label_l = label.lower()
        if isinstance(frame, IdentityReferenceFrame) or name == 'IdentityReferenceFrame':
            return 'world'
        if 'NonRotating' in name or 'non-rotating' in label_l:
            return 'body_centered_non_rotating'
        if 'BodyDirection' in name or 'direction' in label_l:
            return 'body_centered_body_direction'
        return 'custom_frame'

    def _debug_prediction_frame_transform(self, path_points, predictor=None):
        if not getattr(self, 'debug_predictor', False):
            return
        try:
            count = self._points_count(path_points)
            if count <= 0:
                return
            mode = self._prediction_frame_transform_mode()
            active_frame = self._active_frame()
            ref_index = getattr(predictor, 'reference_body_index', None) if predictor is not None else None
            if ref_index is None:
                primary = getattr(active_frame, 'primary_body', None)
                if primary is None:
                    primary = getattr(active_frame, 'target_body', None)
                body_index_by_id = getattr(self, '_current_body_index_by_id', {}) or {}
                ref_index = body_index_by_id.get(id(primary)) if primary is not None else None
            key = (mode, int(count), id(active_frame), ref_index)
            if key == getattr(self, '_prediction_frame_transform_debug_key', None):
                return
            self._prediction_frame_transform_debug_key = key
            if mode == 'world':
                print(f"PRED_DBG_FRAME_TRANSFORM: mode=world points={int(count)}", flush=True)
            else:
                print(
                    "PRED_DBG_FRAME_TRANSFORM: "
                    f"mode={mode} "
                    f"ref_index={ref_index if ref_index is not None else 'n/a'} "
                    f"points={int(count)}",
                    flush=True,
                )
        except Exception:
            pass

    def _reset_reference_trajectories(self):
        self._reference_traj_points = {}
        self._reference_traj_last_sample_time = None
        # frame-wechsel: gecachte kamera-frame-position ist nicht mehr gültig.
        self._camera_frame_xy_key = None

    def _init_gpu_helpers(self):
        """Erstellt wiederverwendbare puffer, programme und VAOs für kritische render-pfade."""
        self._ensure_poly_vbo()
        self._ensure_quad_vbo()
        self._init_line_pipeline()
        self._init_ortho_pipeline()
        self._init_body_pipeline()
        self._init_texquad_pipeline()
        self._init_background_pipeline()

    def _load_shader_source(self, filename):
        path = os.path.join(self._shader_dir, filename)
        with open(path, 'r', encoding='utf-8') as shader_file:
            return shader_file.read()

    def _compile_shader_program(self, vertex_filename, fragment_filename, label):
        """Lädt und linkt ein GLSL-programm; None bei fehler (pipeline degradiert)."""
        try:
            vertex_source = self._load_shader_source(vertex_filename)
            fragment_source = self._load_shader_source(fragment_filename)
            return self.ctx.program(
                vertex_shader=vertex_source,
                fragment_shader=fragment_source,
            )
        except Exception as exc:
            self.debug_info['shader_error'] = f"{label}: {exc}"
            print(f"Shader pipeline fallback ({label}): {exc}")
            return None

    def _init_line_pipeline(self):
        """Linien in top-down-bildschirmkoordinaten (y-flip in line.vert)."""
        program = self._compile_shader_program('line.vert', 'line.frag', 'line')
        if program is None:
            self._line_program = None
            self._line_vao = None
            return

        try:
            self._line_vao = self.ctx.vertex_array(
                program, [(self._poly_vbo, '2f', 'a_pos')]
            )
            self._line_program = program
        except Exception as exc:
            self.debug_info['shader_error'] = f"line: {exc}"
            print(f"Shader pipeline fallback (line): {exc}")
            try:
                program.release()
            except Exception:
                pass
            self._line_program = None
            self._line_vao = None

    def _init_ortho_pipeline(self):
        """Geometrie in der alten fixed-function-ortho-konvention (y nach oben).

        Ersetzt die früheren immediate-mode-pfade unter gluOrtho2D(0, w, 0, h)
        (schiffspfeil, debug-kreuze): exakt dieselbe pixel-abbildung, nur via
        shader (ortho.vert, OHNE den y-flip von line.vert). Der konventions-
        unterschied zwischen line- und ortho-pfad ist absichtlich und
        dokumentiert (CLAUDE.md, render-convention caveat).
        """
        program = self._compile_shader_program('ortho.vert', 'line.frag', 'ortho')
        if program is None:
            self._ortho_program = None
            self._ortho_vao = None
            return

        try:
            self._ortho_vao = self.ctx.vertex_array(
                program, [(self._poly_vbo, '2f', 'a_pos')]
            )
            self._ortho_program = program
        except Exception as exc:
            self.debug_info['shader_error'] = f"ortho: {exc}"
            print(f"Shader pipeline fallback (ortho): {exc}")
            try:
                program.release()
            except Exception:
                pass
            self._ortho_program = None
            self._ortho_vao = None

    def _init_body_pipeline(self):
        program = self._compile_shader_program('body.vert', 'body.frag', 'body')
        if program is None:
            self._body_program = None
            self._body_vao = None
            return

        try:
            self._body_vao = self.ctx.vertex_array(
                program, [(self._ensure_quad_vbo(), '2f', 'a_corner')]
            )
            self._body_program = program
        except Exception as exc:
            self.debug_info['shader_error'] = f"body: {exc}"
            print(f"Shader pipeline fallback (body): {exc}")
            try:
                program.release()
            except Exception:
                pass
            self._body_program = None
            self._body_vao = None

        self._init_body_style_pipeline()
        self._init_body_icon_pipeline()

    def _init_body_icon_pipeline(self):
        """Programm der positions-marke.

        Teilt sich das statische einheits-quad mit der body- und der
        FXAA-pipeline; die marke braucht keine eigene geometrie, weil das
        zellmuster im fragment-shader aufgeloest wird.
        """
        program = self._compile_shader_program(
            'body_icon.vert', 'body_icon.frag', 'body_icon')
        if program is None:
            self._body_icon_program = None
            self._body_icon_vao = None
            return
        try:
            self._body_icon_vao = self.ctx.vertex_array(
                program, [(self._ensure_quad_vbo(), '2f', 'a_corner')]
            )
            self._body_icon_program = program
        except Exception as exc:
            self.debug_info['shader_error'] = f"body_icon: {exc}"
            print(f"Shader pipeline fallback (body_icon): {exc}")
            try:
                program.release()
            except Exception:
                pass
            self._body_icon_program = None
            self._body_icon_vao = None

    def _init_body_style_pipeline(self):
        """Programme fuer die vektor-zeichnung der koerper.

        Anders als die uebrigen pipelines gibt es hier KEIN gemeinsames VAO:
        jeder koerper hat seine eigene geometrie und damit seinen eigenen
        puffer (siehe `_upload_body_style`).
        """
        self._body_surface_program = self._compile_shader_program(
            'body_surface.vert', 'body_surface.frag', 'body_surface')
        self._body_line_program = self._compile_shader_program(
            'body_line.vert', 'body_line.frag', 'body_line')
        if self._body_surface_program is None or self._body_line_program is None:
            self._body_surface_program = None
            self._body_line_program = None

    def _init_texquad_pipeline(self):
        """Texturierte quads (labels, HUD) in der ortho-konvention (y nach oben)."""
        program = self._compile_shader_program('texquad.vert', 'texquad.frag', 'texquad')
        if program is None:
            self._texquad_program = None
            self._texquad_vao = None
            return

        try:
            program['u_texture'].value = 0
            self._texquad_vao = self.ctx.vertex_array(
                program, [(self._ensure_quad_vbo(), '2f', 'a_corner')]
            )
            self._texquad_program = program
        except Exception as exc:
            self.debug_info['shader_error'] = f"texquad: {exc}"
            print(f"Shader pipeline fallback (texquad): {exc}")
            try:
                program.release()
            except Exception:
                pass
            self._texquad_program = None
            self._texquad_vao = None

    def _init_background_pipeline(self):
        """Hintergrund-ebene: vollbild-quad (gitter) + punkt-sprites (sterne).

        Beide programme degradieren einzeln zu None; fehlt eines, zeichnet der
        hintergrund die jeweils andere schicht weiter.
        """
        program = self._compile_shader_program(
            'background.vert', 'background.frag', 'background')
        if program is None:
            self._background_program = None
            self._background_vao = None
        else:
            try:
                self._background_vao = self.ctx.vertex_array(
                    program, [(self._ensure_quad_vbo(), '2f', 'a_pos')]
                )
                self._background_program = program
            except Exception as exc:
                self.debug_info['shader_error'] = f"background: {exc}"
                print(f"Shader pipeline fallback (background): {exc}")
                try:
                    program.release()
                except Exception:
                    pass
                self._background_program = None
                self._background_vao = None

        star = self._compile_shader_program('star.vert', 'star.frag', 'star')
        if star is None:
            self._star_program = None
            self._star_vao = None
            return
        self._star_program = star
        self._star_vao = None      # entsteht beim ersten VBO-schreiben

    def _ensure_star_buffer(self):
        """Laedt die sterntabelle in den instanz-VBO, wenn die dichte wechselt.

        Der puffer ist STATISCH: parallaxe und funkelphase stehen je stern
        darin, drift und zeit sind uniforms. Es wird also nur bei einer
        dichteaenderung geschrieben, nicht je bild.

        Gezeichnet wird als INSTANZIERTES quad, nicht als punkt-sprite --
        `gl_PointCoord` liefert auf dem NVIDIA-treiber dieses rechners
        konstant (0, 0) und liess damit die zellmaske jedes sternfragment
        verwerfen. Begruendung in shaders/star.vert.
        """
        if self._star_program is None:
            return None
        table = self.background.star_table()
        if table is None or table.shape[0] == 0:
            return None
        if not self.background.take_stars_dirty() and self._star_vao is not None:
            return self._star_vao

        data = np.ascontiguousarray(table, dtype='f4')
        if self._star_vbo is not None:
            try:
                self._star_vbo.release()
            except Exception:
                pass
        if self._star_vao is not None:
            try:
                self._star_vao.release()
            except Exception:
                pass
        try:
            if self._star_corner_vbo is None:
                # Einheitsquadrat 0..1 in TRIANGLE_STRIP-reihenfolge, von
                # allen instanzen geteilt.
                corner = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                                  dtype='f4')
                self._star_corner_vbo = self.ctx.buffer(corner.tobytes())
            self._star_vbo = self.ctx.buffer(data.tobytes())
            self._star_vao = self.ctx.vertex_array(
                self._star_program,
                [
                    (self._star_corner_vbo, '2f', 'a_corner'),
                    # '/i' = je INSTANZ ein satz, nicht je vertex.
                    (self._star_vbo, '2f 4f 1f/i',
                     'a_pos', 'a_param', 'a_phase'),
                ],
            )
            self._star_vbo_count = int(data.shape[0])
        except Exception as exc:
            self.debug_info['shader_error'] = f"star buffer: {exc}"
            print(f"Star buffer fallback: {exc}")
            self._star_vao = None
            self._star_vbo = None
            self._star_vbo_count = 0
        return self._star_vao

    def _draw_background(self, camera, real_dt):
        """Zeichnet sternenfeld und gitter -- die unterste schicht.

        Laeuft VOR allem anderen in denselben framebuffer und ersetzt damit
        praktisch den clear (der bleibt trotzdem stehen, falls die ebene
        abgeschaltet oder ein programm ausgefallen ist).

        Die ebene liegt bewusst INNERHALB des FXAA-passes: sie ist die
        unterste schicht, alles andere spaeter herauszuziehen wuerde die
        reihenfolge zerreissen. Bei deckkraeften um 3 % ist der kantenfilter
        hier ohnehin nahe an einer identitaet -- anders als bei text, siehe
        .claude/rules/rendering.md.
        """
        bg = self.background
        if not bg.enabled:
            return

        # Das STERNENFELD haengt an der echten eigenbewegung des verfolgten
        # koerpers (absolut, damit ein rahmenwechsel es nicht ruckt); steht die
        # kamera frei, uebernimmt der schwenk.
        #
        # ACHTUNG: hier stand einmal `focus.velocity`. Das ist fuer
        # himmelskoerper IMMER (0, 0) -- solar_system.json setzt es so, und
        # world.update_planets schreibt nur die kepler-POSITION. Das feld stand
        # damit bei jedem koerper ausser dem Schiff still. Uebergeben wird
        # deshalb die position, abgeleitet wird in background._focus_speed.
        focus = getattr(camera, 'target', None)
        focus_world_xy = None
        focus_frame_xy = None
        if focus is not None:
            position = getattr(focus, 'position', None)
            if position is not None:
                focus_world_xy = (float(position.x), float(position.y))
                focus_frame_xy = self._frame_transform_xy(*focus_world_xy)

        # Das GITTER ist ein festes lattice im aktiven plot-frame. Sein anker
        # ist damit schlicht die kameraposition darin -- der bezugskoerper
        # steht darauf still, mond und schiff wandern darueber, ein schwenk
        # schiebt es genau so weit wie die welt. Der bezugskoerper muss hier
        # nicht gesondert hinein: er STECKT bereits in der frame-transform.
        cam_xy = self._frame_camera_xy(camera)
        grid_target = bg.grid_target_xy(cam_xy, focus_frame_xy)
        # Wogegen der anker gemessen ist. Wechselt der schluessel (R / 1 / 2,
        # oder das blickziel bei anchor="focus"), ist der sprung im ziel kein
        # flug -- die ebene uebernimmt ihn dann, statt ihn abzufahren.
        frame = self._active_frame()
        grid_key = (frame.__class__.__name__,
                    str(getattr(frame, 'label', '')),
                    bg.grid_anchor,
                    getattr(focus, 'name', None)
                    if bg.grid_anchor == "focus" else None)

        bg.update(
            real_dt,
            camera.scale,
            getattr(camera, 'target_scale', camera.scale),
            (float(camera.position.x), float(camera.position.y)),
            focus_world_xy=focus_world_xy,
            # Nur der KOERPER, nicht der rahmen: die sterne rechnen in
            # absoluten weltkoordinaten, ein rahmenwechsel aendert daran
            # nichts und darf die ableitung nicht neu ansetzen.
            focus_key=(id(focus), getattr(focus, 'name', None)),
            sim_time=self._frame_time_s,
            grid_target=grid_target,
            grid_key=grid_key,
            viewport=(self.width, self.height),
        )

        anchor_xy = bg.anchor_xy()

        viewport = (float(self.width), float(self.height))
        accent = bg.accent_rgb()
        # Virtuelle pixelgroesse in DESIGN-einheiten -- wie jede andere
        # UI-groesse, sonst zerfaellt das raster bei anderer aufloesung.
        pixel = max(1.0, self.ui_px(bg.pixel_size))
        pixel_round = min(1.0, max(0.0, float(bg.pixel_round)))

        # ------------------------------------------------- gitter/grundflaeche
        if self._background_program is not None and self._background_vao is not None:
            levels = bg.levels(camera.scale, anchor_xy[0], anchor_xy[1]) \
                if bg.grid_enabled else []
            count = min(len(levels), background.MAX_LEVELS)

            program = self._background_program
            self._set_uniform(program, 'u_viewport', '_background_viewport',
                              viewport)
            self._write_uniform(program, 'u_accent', accent)
            self._write_uniform(program, 'u_grid_opacity', float(bg.grid_opacity))
            self._write_uniform(program, 'u_pixel', pixel)
            self._write_uniform(program, 'u_pixel_round', pixel_round)
            self._write_uniform(program, 'u_level_count', int(count))
            if count:
                # Die uniform-arrays werden IMMER voll geschrieben: ein rest
                # aus dem letzten bild wuerde sonst mitgezeichnet, sobald
                # u_level_count wieder steigt.
                pad = background.MAX_LEVELS - count
                self._write_uniform(program, 'u_level_sp',
                                    [lv.spacing_px for lv in levels[:count]] + [0.0] * pad)
                self._write_uniform(program, 'u_level_alpha',
                                    [lv.alpha for lv in levels[:count]] + [0.0] * pad)
                self._write_uniform(program, 'u_level_node',
                                    [lv.node_alpha for lv in levels[:count]] + [0.0] * pad)
                # ACHTUNG: `u_level_phase` ist ein vec2-ARRAY. moderngl will
                # dafuer eine liste von PAAREN -- eine flache liste wirft
                # "Value after * must be an iterable, not float". Das ist
                # genau der fehler, der hier einmal drin war: der schreib-
                # versuch schlug still fehl, die phasen blieben null, und das
                # gitter klebte am bildschirm statt an der welt.
                phases = [(lv.phase_a, lv.phase_b) for lv in levels[:count]]
                phases.extend([(0.0, 0.0)] * pad)
                self._write_uniform(program, 'u_level_phase', phases)

            # Das quad ueberschreibt jeden pixel -- ohne blending, sonst
            # mischt es sich mit der clear-farbe.
            self.ctx.disable(moderngl.BLEND)
            self._background_vao.render(moderngl.TRIANGLE_STRIP)
            self.ctx.enable(moderngl.BLEND)

        # --------------------------------------------------------- sternenfeld
        if not bg.stars_enabled:
            return
        vao = self._ensure_star_buffer()
        if vao is None or self._star_vbo_count <= 0:
            return

        program = self._star_program
        self._set_uniform(program, 'u_viewport', '_star_viewport', viewport)
        self._write_uniform(program, 'u_pan',
                            (float(bg.star_pan_px[0]), float(bg.star_pan_px[1])))
        self._write_uniform(program, 'u_time', float(bg.time_s))
        self._write_uniform(program, 'u_opacity', float(bg.star_opacity))
        self._write_uniform(program, 'u_star_zoom', float(bg.star_zoom))
        self._write_uniform(program, 'u_zoom_amount', float(bg.zoom_amount()))
        self._write_uniform(program, 'u_pixel', pixel)
        self._write_uniform(program, 'u_pixel_round', pixel_round)

        # Ein quad je stern, alle vier ecken aus demselben puffer. Kein
        # PROGRAM_POINT_SIZE, kein gl_PointCoord -- siehe star.vert.
        vao.render(moderngl.TRIANGLE_STRIP, vertices=4,
                   instances=self._star_vbo_count)

    def _ensure_poly_vbo(self):
        """Geteilter dynamischer vertex-puffer für polylines und ortho-geometrie."""
        if self._poly_vbo is None:
            initial_size = 4096 * 8  # bytes; wächst bei bedarf via orphan()
            self._poly_vbo = self.ctx.buffer(reserve=initial_size, dynamic=True)
            self._poly_vbo_size = initial_size
        return self._poly_vbo

    def _ensure_quad_vbo(self):
        """Statisches einheits-quad (-1..1, TRIANGLE_STRIP-reihenfolge).

        Geteilt von body-, FXAA- und texquad-pipeline.
        """
        if self._quad_vbo is None:
            quad = np.array(
                [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=np.float32
            )
            self._quad_vbo = self.ctx.buffer(quad.tobytes())
        return self._quad_vbo

    def _write_poly_vertices(self, arr):
        """Lädt ein (N,2)-float32-array in den geteilten dynamischen VBO.

        orphan() reallokiert bei bedarf nur den speicher und behält das
        buffer-objekt — die VAOs der line-/ortho-pipeline bleiben gültig.
        """
        self._ensure_poly_vbo()
        data_size = int(arr.nbytes)
        if data_size > int(self._poly_vbo_size):
            self._poly_vbo.orphan(data_size)
            self._poly_vbo_size = data_size
        self._poly_vbo.write(arr)
        return int(arr.shape[0])

    def _draw_polyline(self, run, color=(1.0, 1.0, 1.0, 1.0), width=1.0):
        """Zeichnet eine bildschirm-space polyline (top-down-konvention) via GLSL+VBO."""
        n = len(run)
        if n < 2 or self._line_vao is None:
            return

        try:
            arr = np.asarray(run, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] != 2:
                arr = arr.reshape((-1, 2))
            if not arr.flags['C_CONTIGUOUS']:
                arr = np.ascontiguousarray(arr, dtype=np.float32)
        except Exception:
            return

        n = self._write_poly_vertices(arr)
        self._set_line_width(width)
        self._set_uniform(self._line_program, 'u_viewport', '_line_viewport',
                          (float(self.width), float(self.height)))
        self._set_uniform(self._line_program, 'u_color', '_line_color', (
            float(color[0]), float(color[1]), float(color[2]), float(color[3])
        ))
        self._line_vao.render(moderngl.LINE_STRIP, vertices=n)

    def _draw_line_segments(self, points, color=(1.0, 1.0, 1.0, 1.0), width=1.0):
        """Zeichnet PAARWEISE strecken (GL_LINES) in der top-down-konvention.

        Dieselbe pipeline und dieselbe abbildung wie `_draw_polyline`, nur
        ohne den zwang, dass alle punkte EINEN zug bilden. Damit gehen
        mehrere kleine, unverbundene figuren (etwa alle apsis-rauten einer
        farbe) in einem einzigen draw an die GPU.
        """
        n = len(points)
        if n < 2 or self._line_vao is None:
            return

        try:
            arr = np.asarray(points, dtype=np.float32).reshape((-1, 2))
            if not arr.flags['C_CONTIGUOUS']:
                arr = np.ascontiguousarray(arr, dtype=np.float32)
        except Exception:
            return
        # GL_LINES verbraucht die punkte paarweise; ein einzelner ueberzaehliger
        # punkt wuerde verworfen, hier gar nicht erst hochgeladen.
        if arr.shape[0] % 2:
            arr = arr[:-1]
        if arr.shape[0] < 2:
            return

        n = self._write_poly_vertices(arr)
        self._set_line_width(width)
        self._set_uniform(self._line_program, 'u_viewport', '_line_viewport',
                          (float(self.width), float(self.height)))
        self._set_uniform(self._line_program, 'u_color', '_line_color', (
            float(color[0]), float(color[1]), float(color[2]), float(color[3])
        ))
        self._line_vao.render(moderngl.LINES, vertices=n)

    def _draw_ortho_shape(self, points, color, mode, width=1.0):
        """Zeichnet geometrie in der alten ortho-konvention (y nach oben).

        Ersatz für die früheren immediate-mode-aufrufe unter
        gluOrtho2D(0, w, 0, h): identische pixel-abbildung, nur via shader.
        """
        n = len(points)
        if n < 2 or self._ortho_vao is None:
            return

        try:
            arr = np.asarray(points, dtype=np.float32).reshape((-1, 2))
            if not arr.flags['C_CONTIGUOUS']:
                arr = np.ascontiguousarray(arr, dtype=np.float32)
        except Exception:
            return

        n = self._write_poly_vertices(arr)
        if mode in (moderngl.LINES, moderngl.LINE_STRIP):
            self._set_line_width(width)
        self._set_uniform(self._ortho_program, 'u_viewport', '_ortho_viewport',
                          (float(self.width), float(self.height)))
        self._set_uniform(self._ortho_program, 'u_color', '_ortho_color', (
            float(color[0]), float(color[1]), float(color[2]), float(color[3])
        ))
        self._ortho_vao.render(mode, vertices=n)

    # ---- GL-zustandscache -------------------------------------------------
    #
    # Jedes `program['u_x'].value = ...` und jedes `ctx.line_width = ...` geht
    # als eigener aufruf in den treiber. Der linien-zeichenweg setzt beides
    # bei JEDEM aufruf neu -- gemessen ~300 uniform-schreibvorgaenge je frame,
    # von denen sich die allermeisten gegenueber dem vorigen aufruf gar nicht
    # geaendert haben (u_viewport ist ueber den ganzen frame konstant, u_color
    # ueber ganze gruppen von linien). Der cache haelt nur den zuletzt
    # GESCHRIEBENEN wert; geschrieben wird weiterhin jeder wechsel, die
    # sichtbare ausgabe ist also unveraendert.

    def _set_uniform(self, program, name, cache_attr, value):
        if getattr(self, cache_attr, None) == value:
            return
        try:
            program[name].value = value
        except Exception:
            return
        setattr(self, cache_attr, value)

    def _write_uniform(self, program, name, value):
        """Uniform ohne cache schreiben.

        Gegenstueck zu `_set_uniform`: fuer werte, die sich ohnehin JEDES bild
        aendern (gitterphasen, sterndrift, zeit) waere der vergleich teurer
        als der schreibvorgang.

        Ein fehlschlag wird NICHT verschluckt, sondern einmal je uniform in
        `debug_info` vermerkt und einmal gedruckt. Ein still fehlschlagender
        schreibversuch sieht sonst aus wie ein shader-fehler: der uniform
        behaelt seinen wert (in der GL: null), und man sucht die ursache im
        GLSL statt im aufrufer. Genau so ging einmal `u_level_phase` als
        flache liste statt als liste von paaren durch.
        """
        try:
            program[name].value = value
        except Exception as exc:
            key = f"uniform:{name}"
            if key not in self.debug_info:
                self.debug_info[key] = f"{type(exc).__name__}: {exc}"
                print(f"Uniform write failed ({name}): {exc}")

    def _write_uniform_array(self, program, name, values):
        """Ein uint-array-uniform in EINEM aufruf schreiben.

        Gegenstueck zu `_write_uniform` fuer arrays: `.value` nimmt bei einem
        array keine liste, `.write()` will die rohbytes.
        """
        try:
            program[name].write(struct.pack(f'{len(values)}I', *values))
        except Exception as exc:
            key = f"uniform:{name}"
            if key not in self.debug_info:
                self.debug_info[key] = f"{type(exc).__name__}: {exc}"
                print(f"Uniform array write failed ({name}): {exc}")

    def _set_line_width(self, width):
        width = float(width)
        if self._gl_line_width == width:
            return
        self.ctx.line_width = width
        self._gl_line_width = width

    def _invalidate_gl_state_cache(self):
        """Nach fenster-/kontextwechseln: alles wieder als unbekannt fuehren."""
        self._line_viewport = None
        self._line_color = None
        self._ortho_viewport = None
        self._ortho_color = None
        self._texquad_viewport = None
        self._texquad_color = None
        self._background_viewport = None
        self._star_viewport = None
        # Die marke haengt mit `u_viewport` an der fenstergroesse -- genau der
        # wert, ueber den ein cache sonst stale wuerde.
        self._icon_viewport = None
        self._icon_tier_alpha = None
        self._icon_grid = None
        self._icon_edge = None
        self._icon_gap = None
        self._icon_rim = None
        self._icon_rim_dark = None
        self._icon_shade = None
        self._icon_halo = None
        self._icon_extent = None
        self._icon_radius = None
        self._icon_unit = None
        self._gl_line_width = None

    def _clip_segment_to_rect(self, x0, y0, x1, y1, left, top, right, bottom):
        """
        Liang-Barsky clipping for screen-space line segments.
        Returns (cx0, cy0, cx1, cy1) or None if the segment is fully outside.
        Screen coordinates: x right, y down.
        """
        dx = x1 - x0
        dy = y1 - y0

        u1 = 0.0
        u2 = 1.0

        # Liang-Barsky gegen die vier kanten. Bewusst ohne zwischen-listen/zip:
        # diese funktion läuft pro segment jeder spur-, orbit- und vorhersage-
        # linie und ist damit der meistaufgerufene pro-frame-pfad. die (pi, qi)-
        # paare sind exakt wie zuvor (links, rechts, oben, unten), nur skalar.

        # links: pi = -dx, qi = x0 - left
        pi = -dx
        qi = x0 - left
        if pi == 0.0:
            if qi < 0.0:
                return None
        else:
            t = qi / pi
            if pi < 0.0:
                if t > u2:
                    return None
                if t > u1:
                    u1 = t
            elif t < u1:
                return None
            elif t < u2:
                u2 = t

        # rechts: pi = dx, qi = right - x0
        pi = dx
        qi = right - x0
        if pi == 0.0:
            if qi < 0.0:
                return None
        else:
            t = qi / pi
            if pi < 0.0:
                if t > u2:
                    return None
                if t > u1:
                    u1 = t
            elif t < u1:
                return None
            elif t < u2:
                u2 = t

        # oben: pi = -dy, qi = y0 - top
        pi = -dy
        qi = y0 - top
        if pi == 0.0:
            if qi < 0.0:
                return None
        else:
            t = qi / pi
            if pi < 0.0:
                if t > u2:
                    return None
                if t > u1:
                    u1 = t
            elif t < u1:
                return None
            elif t < u2:
                u2 = t

        # unten: pi = dy, qi = bottom - y0
        pi = dy
        qi = bottom - y0
        if pi == 0.0:
            if qi < 0.0:
                return None
        else:
            t = qi / pi
            if pi < 0.0:
                if t > u2:
                    return None
                if t > u1:
                    u1 = t
            elif t < u1:
                return None
            elif t < u2:
                u2 = t

        return (
            x0 + u1 * dx,
            y0 + u1 * dy,
            x0 + u2 * dx,
            y0 + u2 * dy,
        )

    def _build_clipped_polyline_runs(self, screen_points, margin_px=128.0,
                                     coords=None):
        """
        Converts one logical predictor polyline into multiple visible screen-space runs.
        Important: preserve original segment topology. Never connect visible points
        across an offscreen gap.

        `coords` sind dieselben punkte als (sx, sy)-arrays. Liegen sie vor
        (und ist numba da), laeuft die ganze zustandsmaschine als EIN
        kernel-aufruf -- vorher war das die teuerste einzelne funktion des
        frames (gemessen ~15 ms bei 4000 segmenten, praktisch alles
        Python-schleifen-overhead). Ohne `coords` oder ohne numba bleibt der
        Python-weg darunter, zeichenweise identisch.

        Rueckgabe: liste von ``(n, 2)``-float64-arrays. Der ganze
        linien-zeichenweg rechnet auf arrays weiter; die frueheren listen
        aus (x, y)-tupeln wurden auf dem weg zur GPU ohnehin wieder in
        arrays umgewandelt.

        `screen_points` darf ``None`` sein, WENN `coords` vorliegt -- dann
        sind die spalten die einzige darstellung der punkte und es wird gar
        keine tupel-liste mehr gebaut.
        """
        have_coords = coords is not None and np is not None
        if screen_points is None:
            if not have_coords:
                return []
            point_count = len(coords[0])
        else:
            point_count = len(screen_points)
        if point_count < 2:
            return []

        left = -float(margin_px)
        top = -float(margin_px)
        right = float(self.width) + float(margin_px)
        bottom = float(self.height) + float(margin_px)

        coords_match = have_coords and len(coords[0]) == point_count

        # Aufrufer ohne spalten (bahnlinien, referenz-spuren) bekommen sie
        # hier einmalig -- sonst laufen genau die durch den langsamen
        # Python-klipper, waehrend die vorhersagelinie den kernel nutzt.
        if not coords_match and np is not None and _LINE_KERNELS_OK:
            try:
                arr = np.asarray(screen_points, dtype=np.float64)
                if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] == point_count:
                    coords = (np.ascontiguousarray(arr[:, 0]),
                              np.ascontiguousarray(arr[:, 1]))
                    have_coords = True
                    coords_match = True
            except Exception:
                pass

        if coords_match and _LINE_KERNELS_OK:
            sx, sy = coords
            return self._clipped_runs_from_arrays(
                sx, sy, left, top, right, bottom)

        if screen_points is None:
            # Ohne numba braucht der Python-weg unten die punkte einzeln.
            sx, sy = coords
            screen_points = list(zip(np.asarray(sx).tolist(),
                                     np.asarray(sy).tolist()))

        segment_indices = None
        if have_coords:
            sx, sy = coords
            if coords_match:
                out_left = sx < left
                out_right = sx > right
                out_top = sy < top
                out_bottom = sy > bottom
                trivially_out = (
                    (out_left[:-1] & out_left[1:])
                    | (out_right[:-1] & out_right[1:])
                    | (out_top[:-1] & out_top[1:])
                    | (out_bottom[:-1] & out_bottom[1:])
                )
                segment_indices = np.flatnonzero(~trivially_out)

        runs = []
        run = []

        if segment_indices is None:
            iterator = range(len(screen_points) - 1)
        else:
            iterator = segment_indices

        previous_index = None
        for i in iterator:
            i = int(i)
            # Uebersprungene segmente sind verworfene segmente: der lauf
            # bricht dort ab, sonst wuerde ueber die luecke hinweg verbunden.
            if previous_index is not None and i != previous_index + 1:
                if len(run) >= 2:
                    runs.append(run)
                run = []
            previous_index = i

            x0, y0 = screen_points[i]
            x1, y1 = screen_points[i + 1]

            clipped = self._clip_segment_to_rect(
                float(x0), float(y0),
                float(x1), float(y1),
                left, top, right, bottom
            )

            if clipped is None:
                if len(run) >= 2:
                    runs.append(run)
                run = []
                continue

            cx0, cy0, cx1, cy1 = clipped

            if not run:
                run = [(cx0, cy0), (cx1, cy1)]
                continue

            last_x, last_y = run[-1]
            gap_px = math.hypot(cx0 - last_x, cy0 - last_y)

            if gap_px > 2.0:
                if len(run) >= 2:
                    runs.append(run)
                run = [(cx0, cy0), (cx1, cy1)]
            else:
                run.append((cx1, cy1))

        if len(run) >= 2:
            runs.append(run)

        # Einheitliche rueckgabe mit dem kernel-weg: (n, 2)-arrays.
        if np is None:
            return runs
        return [np.asarray(r, dtype=np.float64) for r in runs]

    def _clipped_runs_from_arrays(self, sx, sy, left, top, right, bottom):
        """Kernel-weg von `_build_clipped_polyline_runs`.

        Ein numba-aufruf statt einer Python-schleife ueber alle segmente;
        die laeufe werden anschliessend nur noch als sichten auf den
        ausgabepuffer herausgeschnitten.
        """
        xs = np.ascontiguousarray(sx, dtype=np.float64)
        ys = np.ascontiguousarray(sy, dtype=np.float64)
        ox, oy, starts, counts = _clip_runs_numba(
            xs, ys, float(left), float(top), float(right), float(bottom))
        if starts.shape[0] == 0:
            return []

        runs = []
        for k in range(starts.shape[0]):
            a = int(starts[k])
            b = a + int(counts[k])
            run = np.empty((b - a, 2), dtype=np.float64)
            run[:, 0] = ox[a:b]
            run[:, 1] = oy[a:b]
            runs.append(run)
        return runs

    def _draw_body_glsl(self, x, y, radius, base_color, atmos_color, atmos_density,
                        light_intensity, light=(0.0, 0.0, 1.0), emissive=1.0,
                        surface_mix=0.0, glow=0.0):
        """Zeichnet einen körper als shader-gesteuertes quad (scheibe + optional atmosphäre + glow).

        `light` ist die richtung ZUR lichtquelle im scheiben-raum (y nach oben);
        `emissive` = 1 schaltet die schattierung ab (stern, positions-icon).
        `surface_mix` verdunkelt die scheibe, sobald die vektor-zeichnung
        darueber liegt -- ohne das leuchtet die volle koerperfarbe durch die
        linien hindurch und die facetten verschwinden.
        """
        if self._body_vao is None:
            return False

        radius_px = max(1.0, float(radius))
        radius_scale = max(0.5, min(2.0, radius_px / 50.0))

        outer_radius = radius_px
        atmos_alpha = 0.0
        atmos_radius = radius_px
        if atmos_density > 0.0:
            # Enger als frueher (war 2.0): mit der neuen kugelschattierung ist
            # der koerper selbst dunkel, und ein halo von zwei radien breite
            # ueberstrahlte dann die halbe bildflaeche.
            atmos_radius = radius_px * 1.22
            outer_radius = max(outer_radius, atmos_radius)
            atmos_alpha = min(float(atmos_density) / 100.0, 1.0) * min(radius_scale, 1.0)

        glow_alpha = 0.0
        if light_intensity > 0.0:
            # Stern: grosser halo. Die alte formel teilte durch 1000 und kam
            # damit auf alpha 4e-4 -- der glow war rechnerisch da und optisch
            # nie zu sehen.
            glow_radius = radius_px * 3.0
            outer_radius = max(outer_radius, glow_radius)
            glow_alpha = min(1.0, 0.22 + float(light_intensity) * 0.30) * radius_scale
        elif glow > 0.0:
            glow_radius = radius_px * 1.28
            outer_radius = max(outer_radius, glow_radius)
            glow_alpha = min(1.0, float(glow))

        core_norm = max(0.001, min(1.0, radius_px / max(outer_radius, 1e-6)))
        if atmos_alpha > 0.0:
            atmos_norm = max(core_norm, min(1.0, atmos_radius / max(outer_radius, 1e-6)))
        else:
            atmos_norm = core_norm

        try:
            prog = self._body_program
            prog['u_center_px'].value = (float(x), float(y))
            prog['u_outer_radius_px'].value = float(outer_radius)
            prog['u_viewport'].value = (float(self.width), float(self.height))
            prog['u_base_color'].value = (
                float(base_color[0]), float(base_color[1]), float(base_color[2])
            )
            prog['u_atmos_color'].value = (
                float(atmos_color[0]), float(atmos_color[1]), float(atmos_color[2])
            )
            prog['u_core_radius_norm'].value = float(core_norm)
            prog['u_atmos_radius_norm'].value = float(atmos_norm)
            prog['u_atmos_alpha'].value = float(atmos_alpha)
            prog['u_glow_alpha'].value = float(glow_alpha)
            prog['u_light'].value = (float(light[0]), float(light[1]), float(light[2]))
            prog['u_ambient'].value = float(self.body_ambient)
            prog['u_emissive'].value = float(emissive)
            prog['u_surface_mix'].value = float(surface_mix)

            self._body_vao.render(moderngl.TRIANGLE_STRIP)
            return True
        except Exception:
            return False

    #: Wie weit das marken-quad ueber den radius hinausreicht (fuer den halo).
    ICON_QUAD_EXTENT = 2.6

    def _body_icon_entry(self, body):
        """Gepacktes zellfeld und farbstufen dieser marke, gecacht.

        Der schluessel ist wie bei `_body_style_key` bewusst NICHT `id(body)`,
        sondern das, was das muster bestimmt: ein neu geladener koerper mit
        denselben angaben bekommt dieselbe marke. Hoechstens ein eintrag je
        koerper und variante, gebaut im hauptthread -- der bau sind ein paar
        dutzend zellen, kein grund fuer einen worker wie bei body_style.
        """
        seed = body_icon.seed_for(body, self.body_icon_seed_offset)
        color = tuple(int(c) for c in tuple(getattr(body, 'color', (255, 255, 255)))[:3])
        key = (seed, str(self.body_icon_variant), int(self.body_icon_grid), color)
        entry = self._body_icon_cache.get(key)
        if entry is None:
            try:
                cells = body_icon.build_icon(
                    seed, self.body_icon_variant, self.body_icon_grid)
                entry = (cells, body_icon.icon_palette(color))
            except Exception as exc:
                # Wie bei body_style: einmal scheitern heisst nie wieder
                # versuchen. Sonst kostet ein kaputter bau jeden frame.
                self.debug_info['body_icon_error'] = f"{type(exc).__name__}: {exc}"
                entry = False
            self._body_icon_cache[key] = entry
        return entry

    def _draw_body_icon(self, body, x, y, radius, r, g, b, fade=1.0):
        """Positions-marke eines körpers, konstanter bildschirmgröße.

        `radius` ist der GEZEICHNETE marken-radius -- siehe
        `_body_icon_draw_radius_px` fuer die skalierung mit dem echten
        koerper; `fade` blendet die marke über dem echten körper aus, siehe
        `_body_icon_fade`.

        Zwei wege. `body_icon_style = "disc"` zeichnet die alte flache scheibe
        über denselben GLSL-körper-shader wie der volle körper: der
        vertex-shader (body.vert) erwartet top-down-screen-koordinaten und
        spiegelt y intern (`ndc.y = 1 - 2*y/h`) — dieselbe konvention wie die
        körper-position. Mit glow/atmosphäre = 0 ergibt der shader
        (core_radius_norm == 1.0) eine flache scheibe in körperfarbe.

        `"pixel"` (voreinstellung) zeichnet statt dessen das gesäte zellmuster
        aus `body_icon.py` — EIN quad, das muster löst der fragment-shader aus
        der icon-lokalen koordinate auf. Es gibt deshalb keine aneinander-
        stossenden primitive und damit keine naht, und weil die koordinate an
        der gleitkomma-position der marke hängt, kann das muster nicht über
        die marke wandern.
        """
        if (self.body_icon_style != "pixel"
                or self._body_icon_program is None
                or self._body_icon_vao is None):
            self._draw_body_glsl(x, y, float(radius), (r, g, b), (r, g, b), 0.0, 0.0)
            return

        entry = self._body_icon_entry(body)
        if not entry:
            self._draw_body_glsl(x, y, float(radius), (r, g, b), (r, g, b), 0.0, 0.0)
            return

        cells, palette = entry
        prog = self._body_icon_program

        # Was ueber alle koerper gleich bleibt, geht ueber den vergleichenden
        # cache; nur position, muster, farbe und ueberblendung je koerper.
        self._set_uniform(prog, 'u_viewport', '_icon_viewport',
                          (float(self.width), float(self.height)))
        self._set_uniform(prog, 'u_tier_alpha', '_icon_tier_alpha',
                          tuple(body_icon.TIER_ALPHA[1:]))
        self._set_uniform(prog, 'u_grid', '_icon_grid', int(cells.grid))
        self._set_uniform(prog, 'u_edge_px', '_icon_edge',
                          float(self.body_icon_edge_px))
        self._set_uniform(prog, 'u_cell_gap', '_icon_gap',
                          float(self.body_icon_cell_gap))
        self._set_uniform(prog, 'u_cell_rim', '_icon_rim',
                          float(self.body_icon_cell_rim))
        self._set_uniform(prog, 'u_cell_rim_dark', '_icon_rim_dark',
                          float(self.body_icon_cell_rim_dark))
        self._set_uniform(prog, 'u_halo_alpha', '_icon_halo',
                          float(self.body_icon_halo_alpha))
        self._set_uniform(prog, 'u_extent', '_icon_extent',
                          float(self.ICON_QUAD_EXTENT))
        self._set_uniform(prog, 'u_radius_px', '_icon_radius', float(radius))
        self._set_uniform(prog, 'u_unit', '_icon_unit', float(cells.unit))

        self._write_uniform(prog, 'u_center_px', (float(x), float(y)))
        # Ein uniform-ARRAY: moderngl schreibt es mit glUniform1uiv, also
        # dicht gepackt -- deshalb .write() statt .value.
        self._write_uniform_array(prog, 'u_cells', cells.words)
        self._write_uniform(prog, 'u_tier_dim', palette[0])
        self._write_uniform(prog, 'u_tier_base', palette[1])
        self._write_uniform(prog, 'u_tier_bright', palette[2])
        self._write_uniform(prog, 'u_fade', float(fade))
        self._write_uniform(prog, 'u_seed', int(cells.seed) & 0xFFFFFFFF)
        self._set_uniform(prog, 'u_shade', '_icon_shade',
                          float(self.body_icon_shade_jitter))

        self._body_icon_vao.render(moderngl.TRIANGLE_STRIP)

    def _update_icon_radius_range(self, bodies):
        """Die spanne der PHYSISCHEN koerper-radien im geladenen system.

        Einmal je frame aus der echten koerperliste bestimmt (28 koerper,
        eine schleife -- kostet nichts), NICHT aus einer festen konstante:
        so passt sich die skalierung automatisch an, welches system gerade
        geladen ist, statt eine zahl aus DIESEM sonnensystem im code zu
        verstecken. Das schiff zaehlt nicht mit -- sein `radius` ist ein
        rein technischer platzhalter (1.0 m), keine physische groesse.
        """
        lo = hi = None
        for body in bodies:
            if getattr(body, 'is_ship', False):
                continue
            r = float(getattr(body, 'radius', 0.0))
            if r <= 0.0:
                continue
            if lo is None or r < lo:
                lo = r
            if hi is None or r > hi:
                hi = r
        if lo is None:
            lo = hi = 1.0
        self._icon_radius_range_m = (lo, hi)

    def _body_icon_size_factor(self, body_radius_m):
        """0..1: wo dieser koerper-radius innerhalb der GELADENEN spanne liegt.

        LOG-skaliert: planeten- und mond-radien liegen ueber mehrere
        groessenordnungen (in `solar_system.json` von Mimas' 2.0e5 m bis
        Sonnes 7.0e8 m -- 3.5 dekaden). Linear interpoliert wuerde alles
        ausser der Sonne auf denselben punkt nahe 0 zusammendruecken.
        """
        lo_m, hi_m = self._icon_radius_range_m
        if hi_m <= lo_m:
            return 0.0
        r = max(lo_m, min(hi_m, float(body_radius_m)))
        return (math.log10(r) - math.log10(lo_m)) / (math.log10(hi_m) - math.log10(lo_m))

    def _body_icon_draw_radius_px(self, body_radius_m):
        """Der GEZEICHNETE radius der marke -- ein je koerper KONSTANTER wert
        aus seinem PHYSISCHEN radius, unabhaengig vom zoom.

        > **Bewusst nicht aus dem aktuellen bildschirmradius abgeleitet --
        > das war die erste, falsche fassung.** `true_radius_px` schrumpft mit
        > jedem herauszoomen gegen null, und genau dort, wo ein koerper zur
        > marke wird, liegt er fast immer weit unter `body_icon_min_radius_px`
        > -- eine mischung `min + (true - min) * einfluss` klemmte deshalb bei
        > JEDEM einfluss-wert exakt auf `min` zurueck, weil `true - min`
        > negativ blieb. Der regler hatte dadurch im spiel keine sichtbare
        > wirkung, obwohl er in einem test mit handgesetzten grossen radien
        > (bewusst weit ueber `min`) korrekt aussah. Die groesse haengt jetzt
        > an `body.radius` selbst -- der bleibt bei jedem zoom derselbe, ein
        > Jupiter-aehnlicher koerper ist also IMMER sichtbar groesser als ein
        > kleiner mond, nicht nur kurz waehrend der ueberblendung.

        `body_icon_size_influence` (0..1) mischt zwischen "immer
        `body_icon_min_radius_px`" (0 -- jede marke gleich gross) und "voll
        nach dem log-skalierten koerper-radius, bis `body_icon_max_radius_px`"
        (1).
        """
        lo = float(self.body_icon_min_radius_px)
        hi = max(lo, float(self.body_icon_max_radius_px))
        influence = max(0.0, min(1.0, float(self.body_icon_size_influence)))
        if influence <= 0.0:
            return lo
        factor = self._body_icon_size_factor(body_radius_m)
        return lo + (hi - lo) * factor * influence

    def _body_icon_fade(self, true_radius_px):
        """Deckkraft der marke bei diesem echten bildschirmradius.

        1.0 unterhalb der schwelle, dann linear auf 0 bis
        `body_icon_min_radius_px * body_icon_fade_factor`. Der echte koerper
        wird in diesem band ganz normal gezeichnet und die marke darueber
        ausgeblendet -- das ist die ueberblendung, und sie kostet den
        koerper-zeichenweg keine zeile.
        """
        lo = float(self.body_icon_min_radius_px)
        hi = lo * float(self.body_icon_fade_factor)
        if true_radius_px < lo:
            return 1.0
        if hi <= lo or true_radius_px >= hi:
            return 0.0
        return 1.0 - (float(true_radius_px) - lo) / (hi - lo)

    # ------------------------------------------------------------------
    # Prozedurale vektor-optik der koerper (D2)
    # ------------------------------------------------------------------

    def _body_style_key(self, body, detail):
        """Cache-schluessel: alles, was die zeichnung bestimmt.

        Bewusst NICHT `id(body)`: der schluessel soll einen neu geladenen
        koerper mit denselben angaben auf dieselbe zeichnung fuehren.
        """
        seed = getattr(body, 'style_seed', None)
        if seed is None:
            seed = body_style.seed_from_name(getattr(body, 'name', '?'))
        mode = getattr(body, 'style_mode', None) or body_style.DEFAULT_MODE
        shape = getattr(body, 'style_shape', None) or body_style.DEFAULT_SHAPE
        color = tuple(int(c) for c in tuple(getattr(body, 'color', (255, 255, 255)))[:3])
        return (int(seed) & 0xFFFFFFFF, str(mode), str(shape), color,
                str(detail), float(self.body_vector_shape_density))

    def _body_detail_levels(self, radius_px):
        """[(stufe, gewicht), ...] fuer diesen bildschirmradius.

        Die stufe, deren facetten am naechsten an `body_vector_facet_px`
        liegen, gewinnt; in einem band um den wechsel herum laufen ZWEI
        stufen mit summe 1 -- das ist die ueberblendung. Gerechnet wird in
        log-groesse, weil die stufen sich in der facettenbreite jeweils
        halbieren, also geometrisch und nicht linear liegen.
        """
        forced = self.body_vector_detail
        levels = body_style.DETAIL_LEVELS
        if forced:
            return ((str(forced), 1.0),)

        radius_px = max(1e-3, float(radius_px))
        target = max(1.0, float(self.body_vector_facet_px))
        blend = max(1e-3, min(0.9, float(self.body_vector_detail_blend)))

        position = 0.0
        for index in range(len(levels) - 1):
            # Bildschirmradius, bei dem stufe index+1 dieselbe facettenbreite
            # traefe wie das ziel.
            switch = target / body_style.FACET_FRACTION[levels[index + 1]]
            low = math.log(switch / (1.0 + blend))
            high = math.log(switch * (1.0 + blend))
            position += max(0.0, min(1.0,
                                     (math.log(radius_px) - low) / (high - low)))

        base = int(math.floor(position))
        frac = position - base
        if base >= len(levels) - 1:
            return ((levels[-1], 1.0),)
        if frac <= 1e-3:
            return ((levels[base], 1.0),)
        return ((levels[base], 1.0 - frac), (levels[base + 1], frac))

    def _body_style_entry(self, body, detail):
        """Gebaute + hochgeladene zeichnung eines koerpers, oder None.

        None heisst 'diesen frame noch nicht' -- entweder ist das budget
        aufgebraucht (dann kommt sie im naechsten frame) oder der bau ist
        fehlgeschlagen (dann nie wieder, der fehler steht in debug_info).
        """
        if not self.body_vector_style or self._body_surface_program is None:
            return None
        key = self._body_style_key(body, detail)
        entry = self._body_style_gpu.get(key)
        if entry is not None:
            return entry or None
        job = self._body_style_jobs.get(key)
        if job is not None:
            if not job.done():
                return None
            del self._body_style_jobs[key]
            return self._finish_body_style(key, body, job.result)

        if len(self._body_style_jobs) >= int(self._body_style_build_budget):
            return None
        executor = self._body_style_executor
        if executor is None:
            try:
                executor = ThreadPoolExecutor(max_workers=1,
                                              thread_name_prefix='bodystyle')
                self._body_style_executor = executor
            except Exception:
                executor = None
        args = (key[0],)
        kwargs = dict(color=key[3], mode=key[1], shape=key[2],
                      coverage=float(self.body_vector_coverage),
                      detail=key[4], shape_density=key[5])
        if executor is None:
            # Ohne threads lieber einen ruckler als gar keine zeichnung.
            return self._finish_body_style(
                key, body, lambda: body_style.build_planet_style(*args, **kwargs))
        self._body_style_jobs[key] = executor.submit(
            body_style.build_planet_style, *args, **kwargs)
        return None

    def _finish_body_style(self, key, body, produce):
        """Ergebnis eines baus in GL-puffer legen. Laeuft im hauptthread."""
        try:
            entry = self._upload_body_style(produce())
        except Exception as exc:
            self.debug_info['body_style_error'] = f"{getattr(body, 'name', '?')}: {exc}"
            print(f"Body style build failed ({getattr(body, 'name', '?')}): {exc}")
            entry = False
        self._body_style_gpu[key] = entry
        return entry or None

    def _upload_body_style(self, style):
        """PlanetStyle -> GL-puffer.

        Die linien werden GETRENNT expandiert: `expand_segments` wirft
        entartete segmente weg, und danach waere die grenze zwischen den
        segmenten unter und ueber den fuellungen nicht mehr bekannt. Diese
        reihenfolge ist nicht kosmetisch -- alphas addieren sich, das
        gitternetz gehoert unter die fuellungen.
        """
        tri = np.ascontiguousarray(style.tri, dtype='f4')
        under = body_style.expand_segments(style.seg[:style.under_segments])
        over = body_style.expand_segments(style.seg[style.under_segments:])
        lines = np.ascontiguousarray(np.concatenate([under, over], axis=0), dtype='f4')

        tri_vbo = self.ctx.buffer(tri.tobytes()) if tri.shape[0] else None
        line_vbo = self.ctx.buffer(lines.tobytes()) if lines.shape[0] else None

        tri_vao = None
        if tri_vbo is not None:
            tri_vao = self.ctx.vertex_array(
                self._body_surface_program,
                [(tri_vbo, '2f 3f 3f 1f 1f',
                  'a_pos', 'a_nrm', 'a_col', 'a_alpha', 'a_dark')],
            )
        line_vao = None
        if line_vbo is not None:
            line_vao = self.ctx.vertex_array(
                self._body_line_program,
                [(line_vbo, '2f 3f 3f 1f 1f 2f 1f 1f 1f',
                  'a_pos', 'a_nrm', 'a_col', 'a_alpha', 'a_dark',
                  'a_dir', 'a_side', 'a_ext', 'a_half')],
            )
        return {
            'tri_vao': tri_vao,
            'tri_count': int(tri.shape[0]),
            'line_vao': line_vao,
            'under_count': int(under.shape[0]),
            'over_count': int(over.shape[0]),
            'buffers': [b for b in (tri_vbo, line_vbo) if b is not None],
            'style': style,
        }

    def _body_detail_fade(self, radius_px):
        """0 unter der schwelle, 1 ab voller groesse, dazwischen linear."""
        lo = float(self.body_vector_min_radius_px)
        hi = max(lo + 1e-6, float(self.body_vector_full_radius_px))
        return max(0.0, min(1.0, (float(radius_px) - lo) / (hi - lo)))

    def _body_light_dir(self, body, x, y):
        """Richtung zur lichtquelle im scheiben-raum, plus emissiv-flag.

        Die richtung wird im BILDSCHIRM gemessen, nicht in weltkoordinaten:
        so folgt die beleuchtung automatisch jedem rotierenden plotting-frame.
        z bleibt 0, das licht liegt also in der bahnebene -- genau das ergibt
        von oben auf das system gesehen die richtige phase.
        """
        source = self._light_screen_xy
        if (not self.body_light_enabled or source is None
                or body is self._light_source_body):
            return (0.0, 0.0, 1.0), 1.0
        dx = float(source[0]) - float(x)
        dy = float(source[1]) - float(y)
        length = math.hypot(dx, dy)
        if length < 1e-9:
            return (0.0, 0.0, 1.0), 1.0
        tilt = max(0.0, min(1.0, float(self.body_light_tilt)))
        plane = math.sqrt(max(0.0, 1.0 - tilt * tilt)) / length
        # bildschirm zaehlt y nach unten, die scheibe nach oben
        return (dx * plane, -dy * plane, tilt), 0.0

    def _draw_body_vector(self, entry, x, y, radius_px, light, emissive, fade):
        """Zeichnet die vektor-zeichnung eines koerpers: drei draw-calls.

        Gitternetz -> fuellungen -> konturen/figuren/ringe, in genau dieser
        reihenfolge (siehe `_upload_body_style`).
        """
        prog_surface = self._body_surface_program
        prog_line = self._body_line_program
        if prog_surface is None or prog_line is None:
            return False
        try:
            for prog in (prog_surface, prog_line):
                prog['u_center_px'].value = (float(x), float(y))
                prog['u_radius_px'].value = float(radius_px)
                prog['u_viewport'].value = (float(self.width), float(self.height))
                prog['u_light'].value = (float(light[0]), float(light[1]), float(light[2]))
                prog['u_light_exp'].value = float(self.body_light_exponent)
                prog['u_fade'].value = float(fade)
                prog['u_emissive'].value = float(emissive)

            line_vao = entry.get('line_vao')
            under = int(entry.get('under_count', 0))
            over = int(entry.get('over_count', 0))
            if line_vao is not None and under > 0:
                line_vao.render(moderngl.TRIANGLES, vertices=under, first=0)
            tri_vao = entry.get('tri_vao')
            if tri_vao is not None and entry.get('tri_count', 0) > 0:
                tri_vao.render(moderngl.TRIANGLES)
            if line_vao is not None and over > 0:
                line_vao.render(moderngl.TRIANGLES, vertices=over, first=under)
            return True
        except Exception as exc:
            self.debug_info['body_style_error'] = f"draw: {exc}"
            return False

    def _recompute_ui_scale(self):
        """Leitet ui_scale aus der fensterhöhe ab. Gibt True bei änderung zurück.

        Skaliert nach höhe (nicht nach fläche oder breite): breite fenster
        sollen die oberfläche nicht aufblähen, hohe fenster schon. Das ist die
        übliche konvention für auflösungsunabhängige spiel-UIs.
        """
        reference = max(float(getattr(self, 'ui_scale_reference_height', 1000.0)), 1.0)
        raw = float(self.height) / reference
        lo = float(getattr(self, 'ui_scale_min', 1.0))
        hi = float(getattr(self, 'ui_scale_max', 3.0))
        scale = max(lo, min(hi, raw)) * float(getattr(self, 'ui_scale_user', 1.0))
        scale = max(0.1, scale)

        previous = getattr(self, 'ui_scale', None)
        self.ui_scale = scale
        # Winzige schwankungen ignorieren: ein neuaufbau aller fonts und ein
        # leeren des textur-caches pro pixel fenster-höhe wäre verschwendung.
        return previous is None or abs(scale - previous) > 1e-3

    def ui_px(self, design_units):
        """Design-einheiten -> tatsächliche pixel bei aktueller ui_scale."""
        return float(design_units) * self.ui_scale

    def _rebuild_fonts(self):
        """Erzeugt die HUD-fonts in der aktuell skalierten pixelgröße neu.

        Neu rastern statt die fertigen texturen zu strecken: gestreckte
        glyphen werden beim hochskalieren unscharf. Der label-textur-cache
        wird dabei ungültig (er ist nach schrifthöhe verschlüsselt) und muss
        geleert werden.
        """
        small_px = max(6, int(round(self.ui_px(self.hud_font_size_small))))
        medium_px = max(6, int(round(self.ui_px(self.hud_font_size_medium))))
        try:
            pygame.font.init()
            self.font_small = pygame.font.SysFont(None, small_px)
            self.font_medium = pygame.font.SysFont(None, medium_px)
        except Exception as exc:
            print(f"RENDERER WARNING: HUD-fonts konnten nicht erzeugt werden ({exc})")
            return
        self._clear_text_caches()

    def _clear_text_caches(self):
        """Gibt alle text-abhängigen GL-texturen frei (font- oder größenwechsel)."""
        for entry in list(getattr(self, '_label_texture_cache', {}).values()):
            texture = entry[0]
            if texture is not None:
                try:
                    texture.release()
                except Exception:
                    pass
        self._label_texture_cache = {}
        # Der pool haelt texturen der ALTEN schriftgroesse -- nach einem
        # font-wechsel passt keine davon mehr, also mit weg.
        for bucket in getattr(self, '_label_texture_pool', {}).values():
            for texture in bucket:
                try:
                    texture.release()
                except Exception:
                    pass
        self._label_texture_pool = {}
        self._label_texture_pool_count = 0
        self._hud_line_surface_cache = {}
        self._hud_cache_key = None

    def set_hud_font_sizes(self, small=None, medium=None):
        """Setzt die DESIGN-schriftgrößen und baut die fonts neu auf."""
        if small is not None:
            self.hud_font_size_small = int(small)
        if medium is not None:
            self.hud_font_size_medium = int(medium)
        self._rebuild_fonts()

    def set_ui_scale_user(self, factor):
        """Benutzer-skalenfaktor (multiplikativ auf die automatische skala)."""
        self.ui_scale_user = max(0.1, float(factor))
        if self._recompute_ui_scale():
            self._rebuild_fonts()

    # Deckel des texturen-recyclings, siehe _acquire_label_texture.
    _LABEL_TEXTURE_POOL_MAX = 64

    def _acquire_label_texture(self, size, data):
        """Beschriftungs-textur besorgen -- moeglichst eine wiederverwendete.

        Wie in ui/text.py: der teure teil einer neuen beschriftung ist die
        GL-allokation, nicht das rastern. Die apsis-labels tragen eine
        entfernung im text und wechseln damit in JEDEM frame; ihre
        verdraengten texturen haben aber immer wieder dieselbe groesse und
        werden deshalb eingesammelt und per `write()` neu befuellt.
        """
        bucket = self._label_texture_pool.get(size)
        if bucket:
            texture = bucket.pop()
            self._label_texture_pool_count -= 1
            try:
                texture.write(data)
                texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
                return texture
            except Exception:
                try:
                    texture.release()
                except Exception:
                    pass
        texture = self.ctx.texture(size, 4, data)
        texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        return texture

    def _retire_label_texture(self, texture, size):
        if texture is None:
            return
        if self._label_texture_pool_count < self._LABEL_TEXTURE_POOL_MAX:
            self._label_texture_pool.setdefault(size, []).append(texture)
            self._label_texture_pool_count += 1
            return
        try:
            texture.release()
        except Exception:
            pass

    def _get_label_texture(self, text, font):
        key = (text, font.get_height())
        entry = self._label_texture_cache.get(key)
        if entry:
            return entry  # (texture, w, h)
        try:
            surface = font.render(text, True, (255, 255, 255))
            texture_data = pygame.image.tostring(surface, 'RGBA', True)
            w, h = surface.get_size()
            # cache deckeln (FIFO): ständig wechselnde texte (speed-label)
            # würden sonst unbegrenzt GL-texturen anhäufen. Stabile labels
            # (körpernamen) werden nach einer eviction einfach neu erzeugt.
            # Die verdraengten TEXTUREN wandern in den pool, statt dass
            # jedes frame neue angelegt werden.
            if len(self._label_texture_cache) >= self._label_texture_cache_max:
                evict_n = max(1, self._label_texture_cache_max // 4)
                for old_key in list(self._label_texture_cache.keys())[:evict_n]:
                    old = self._label_texture_cache.pop(old_key)
                    self._retire_label_texture(old[0], (int(old[1]), int(old[2])))
            texture = self._acquire_label_texture((w, h), texture_data)
            self._label_texture_cache[key] = (texture, w, h)
            return (texture, w, h)
        except Exception:
            return None

    def _draw_texture_ortho(self, texture, x, y, width, height, color=(1.0, 1.0, 1.0, 1.0)):
        """Zeichnet eine textur als quad in der ortho-konvention (y nach oben).

        Ersatz für die früheren immediate-mode glTexCoord/glVertex-quads unter
        gluOrtho2D(0, w, 0, h): (x, y) ist die untere linke ecke, texcoord
        (0, 0) liegt ebendort (texturen werden vertikal geflippt hochgeladen).

        color toent die textur multiplikativ (texquad.frag, u_color). Der
        uniform MUSS gesetzt werden -- GL initialisiert uniforms mit 0, ein
        ausgelassenes u_color zeichnet also nichts.
        """
        if self._texquad_vao is None or texture is None:
            return
        # AUF DAS PIXELRASTER RASTEN. Die weltabgeleiteten label-positionen
        # sind subpixelgenau (Erde z. B. bei y=113.7048). Bei LINEAR-filterung
        # verteilt ein solcher versatz jede glyphenzeile auf ZWEI pixelzeilen:
        # der text wird weich und bekommt eine schwache kopie darueber/darunter
        # -- sieht aus wie eine zweite zahl unter der zahl. Gemessen faellt der
        # anteil voll deckender pixel von 19.5 % auf 9 %.
        # Das HUD war nie betroffen, weil es ganzzahlige ursprungswerte nutzt.
        # Die textur wird 1:1 gezeichnet, deshalb genuegt das runden der ecke.
        self._texquad_program['u_rect'].value = (
            round(float(x)), round(float(y)), float(width), float(height)
        )
        # u_viewport ist ueber den ganzen frame konstant, u_color ueber
        # ganze gruppen von beschriftungen -- nur wechsel schreiben.
        self._set_uniform(self._texquad_program, 'u_viewport',
                          '_texquad_viewport',
                          (float(self.width), float(self.height)))
        self._set_uniform(self._texquad_program, 'u_color', '_texquad_color', (
            float(color[0]), float(color[1]), float(color[2]), float(color[3])
        ))
        texture.use(location=0)
        self._texquad_vao.render(moderngl.TRIANGLE_STRIP)

    def _ortho_y(self, y_topdown):
        """Top-down bildschirm-Y (wie _world_to_screen_xy liefert) -> ortho-Y.

        Die welt wird top-down gezeichnet (line.vert flippt y), text und
        schiffs-pfeil laufen dagegen ueber die ortho-konvention (y nach oben,
        ursprung unten links). Ohne diese umrechnung landet alles, was aus
        weltkoordinaten kommt, an der ueber die BILDSCHIRMMITTE gespiegelten
        position -- unsichtbar solange das objekt genau mittig steht, und mit
        wachsendem abstand zur mitte immer weiter daneben.
        """
        return float(self.height) - float(y_topdown)

    def _blit_text_topdown(self, text, x_left, y_top, font, color=(1.0, 1.0, 1.0, 1.0)):
        """Text an TOP-DOWN koordinaten zeichnen (x = links, y = oberkante).

        Nimmt dem aufrufer die ortho-umrechnung ab: _draw_texture_ortho
        erwartet die UNTERE linke ecke in ortho-Y. `color` toent multiplikativ
        (texquad.frag) -- der alphakanal blendet den text aus.
        """
        entry = self._get_label_texture(text, font)
        text_h = float(entry[2]) if entry else float(font.get_height())
        self._blit_cached_text(text, x_left, self._ortho_y(y_top) - text_h, font,
                               color=color)

    def _blit_cached_text(self, text, x, y, font, color=(1.0, 1.0, 1.0, 1.0)):
        entry = self._get_label_texture(text, font)
        if not entry:
            # fallback: one-shot-textur ohne cache erzeugen, zeichnen, freigeben
            try:
                surface = font.render(text, True, (255, 255, 255))
                texture_data = pygame.image.tostring(surface, 'RGBA', True)
                w, h = surface.get_size()
                texture = self.ctx.texture((w, h), 4, texture_data)
                texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
                self._draw_texture_ortho(texture, x, y, w, h, color=color)
                texture.release()
            except Exception:
                pass
            return
        texture, w, h = entry
        self._draw_texture_ortho(texture, x, y, w, h, color=color)

    def _record_reference_trajectories(self, bodies):
        if not self.reference_trajectories_enabled:
            return

        sample_step = max(0.0, float(self.reference_trajectories_sample_step_s))
        if self._reference_traj_last_sample_time is not None and sample_step > 0.0:
            if abs(float(self._frame_time_s) - float(self._reference_traj_last_sample_time)) < sample_step:
                return

        active_ids = set()
        for body in bodies:
            if getattr(body, 'is_ship', False):
                continue

            body_id = id(body)
            active_ids.add(body_id)
            trail = self._reference_traj_points.get(body_id)
            if trail is None:
                # Fester numpy-puffer statt deque von tupeln: das zeichnen
                # braucht die spur als array, und np.asarray über eine
                # tupel-liste kostete pro körper und frame spürbar zeit
                # (27 körper x bis zu 300 punkte, jeden frame neu gewandelt).
                cap = max(64, int(self.reference_trajectories_max_points))
                trail = {'buf': np.empty((cap, 2), dtype=np.float64), 'n': 0}
                self._reference_traj_points[body_id] = trail

            try:
                fx, fy = self._frame_transform_xy(float(body.position.x), float(body.position.y))
            except Exception:
                continue

            buf = trail['buf']
            n = trail['n']
            if n > 0:
                dx = fx - buf[n - 1, 0]
                dy = fy - buf[n - 1, 1]
                if dx * dx + dy * dy < 1e-18:
                    continue
            if n < buf.shape[0]:
                buf[n, 0] = fx
                buf[n, 1] = fy
                trail['n'] = n + 1
            else:
                buf[:-1] = buf[1:]
                buf[-1, 0] = fx
                buf[-1, 1] = fy

        stale_ids = [k for k in self._reference_traj_points.keys() if k not in active_ids]
        for stale_id in stale_ids:
            del self._reference_traj_points[stale_id]

        self._reference_traj_last_sample_time = float(self._frame_time_s)

    def _draw_reference_trajectories(self, bodies, camera):
        if not self.reference_trajectories_enabled:
            return

        camera_frame_xy = self._frame_camera_xy(camera)
        half_w = self.width * 0.5
        half_h = self.height * 0.5
        scale = float(camera.scale)

        for body in bodies:
            if getattr(body, 'is_ship', False):
                continue

            trail = self._reference_traj_points.get(id(body))
            if trail is None or trail['n'] < 2:
                continue

            # Vektorisiert statt python-schleife: spuren haben bis zu
            # reference_trajectories_max_points punkte pro körper und frame.
            # `buf[:n]` ist eine sicht auf den ringpuffer -- keine wandlung.
            arr = trail['buf'][:trail['n']]
            sxs = half_w + (arr[:, 0] - camera_frame_xy[0]) * scale
            sys_ = half_h - (arr[:, 1] - camera_frame_xy[1]) * scale
            min_sx = float(sxs.min()); max_sx = float(sxs.max())
            min_sy = float(sys_.min()); max_sy = float(sys_.max())

            # Größen-schwelle: kollabiert die ganze spur auf eine sub-pixel-fläche
            # (z. B. weit herausgezoomt), ist sie ohnehin unsichtbar -> nicht
            # zeichnen. Die position des körpers zeigt dann sein icon.
            min_px = float(self.reference_traj_min_screen_px)
            if (max_sx - min_sx) < min_px and (max_sy - min_sy) < min_px:
                continue

            # Komplett off-screen liegende spur: weder punkte-liste bauen noch
            # pro segment clippen.
            margin = float(self.prediction_visibility_margin_px)
            right = self.width + margin
            bottom = self.height + margin
            if max_sx < -margin or min_sx > right or max_sy < -margin or min_sy > bottom:
                continue

            base = getattr(body, 'color', (200, 200, 200))
            cr = min(1.0, max(0.0, base[0] / 255.0))
            cg = min(1.0, max(0.0, base[1] / 255.0))
            cb = min(1.0, max(0.0, base[2] / 255.0))

            # ALS SPALTEN WEITERREICHEN. Die punkte liegen schon als arrays
            # vor; die tupel-liste, die hier stand, wurde vom klipper und
            # von _draw_polyline sofort wieder in arrays zurueckverwandelt
            # -- bei bis zu 300 punkten je koerper und frame reine arbeit
            # ohne ergebnis.
            screen_points = np.empty((sxs.shape[0], 2), dtype=np.float64)
            screen_points[:, 0] = sxs
            screen_points[:, 1] = sys_

            if min_sx >= -margin and max_sx <= right and min_sy >= -margin and max_sy <= bottom:
                # Spur liegt vollständig im sichtfenster: Liang-Barsky wäre für
                # jedes segment ein no-op und lieferte exakt einen run.
                runs = (screen_points,)
            else:
                runs = self._visible_window_runs(screen_points, margin_px=margin,
                                                 coords=(sxs, sys_))
            for run in runs:
                if len(run) < 2:
                    continue
                self._draw_polyline(run, color=(cr, cg, cb, 0.42), width=1.0)

    # ------------------------------------------------------------------
    # Bahnlinien der koerper (rechnung in orbit_lines.py)
    # ------------------------------------------------------------------

    def _ensure_orbit_line_set(self):
        """Der zustandsbehaftete teil, ueber frames hinweg gehalten.

        Die konfiguration wird JE FRAME hineingeschrieben statt das objekt
        neu zu bauen -- ein neubau wuerde die eingeblendeten deckkraefte
        verwerfen, und dann blitzt beim drehen an einem regler die ganze
        szene auf.
        """
        oset = self._orbit_line_set
        if oset is None:
            oset = orbit_lines.OrbitLineSet()
            self._orbit_line_set = oset
        oset.track_samples = max(8, int(self.orbit_line_track_samples))
        oset.soi_full = float(self.orbit_line_soi_full)
        oset.soi_fade = float(self.orbit_line_soi_fade)
        oset.reveal_full = float(self.orbit_line_reveal_full)
        oset.reveal_fade = float(self.orbit_line_reveal_fade)
        oset.alpha_max = float(self.orbit_line_alpha_max)
        oset.alpha_floor = float(self.orbit_line_alpha_floor)
        oset.alpha_floor_focus = float(self.orbit_line_alpha_floor_focus)
        oset.fade_rate = float(self.orbit_line_fade_rate)
        oset.full_orbit_enabled = bool(self.orbit_line_full_orbit_enabled)
        oset.full_samples = max(16, int(self.orbit_line_full_samples))
        oset.full_max_span_s = float(self.orbit_line_full_max_span_s)
        return oset

    def _draw_frame_polyline(self, screen_x, screen_y, color, width,
                             min_screen_px=0.0):
        """Fertig projizierte bildschirmpunkte als polylinie, mit culling."""
        n = int(screen_x.shape[0])
        if n < 2:
            return False
        if not (np.all(np.isfinite(screen_x)) and np.all(np.isfinite(screen_y))):
            return False

        min_sx = float(screen_x.min()); max_sx = float(screen_x.max())
        min_sy = float(screen_y.min()); max_sy = float(screen_y.max())

        margin = float(self.prediction_visibility_margin_px)
        right = self.width + margin
        bottom = self.height + margin
        if max_sx < -margin or min_sx > right or max_sy < -margin or min_sy > bottom:
            return False
        if (max_sx - min_sx) < min_screen_px and (max_sy - min_sy) < min_screen_px:
            return False

        pts = np.empty((n, 2), dtype=np.float64)
        pts[:, 0] = screen_x
        pts[:, 1] = screen_y

        if min_sx >= -margin and max_sx <= right and min_sy >= -margin and max_sy <= bottom:
            runs = (pts,)
        else:
            runs = self._build_clipped_polyline_runs(
                pts, margin_px=margin, coords=(screen_x, screen_y))

        drew = False
        for run in runs:
            if len(run) >= 2:
                self._draw_polyline(run, color=color, width=width)
                drew = True
        return drew

    def _draw_end_cap(self, sx, sy, color, size_px):
        """Kleine raute auf dem endpunkt einer linie.

        Die endkappen sind der eigentliche messwert der ganzen funktion:
        liegt die des koerpers auf der des schiffs, ist das schiff zur
        endzeit der vorhersage dort, wo der koerper dann steht.
        """
        if not (math.isfinite(sx) and math.isfinite(sy)):
            return
        if sx < -size_px or sx > self.width + size_px:
            return
        if sy < -size_px or sy > self.height + size_px:
            return
        r = float(size_px)
        diamond = [(sx, sy - r), (sx + r, sy), (sx, sy + r), (sx - r, sy),
                   (sx, sy - r)]
        self._draw_polyline(diamond, color=color, width=1.0)

    def _draw_body_disc_outline(self, sx, sy, r_px, color):
        """Kreis-umriss mit dem ECHTEN radius des koerpers auf dem linienende.

        Das ist der messwert der bahnlinie: liegt dieser kreis ueber der
        weissen schiffs-endkappe, steckt das schiff zur endzeit der vorhersage
        im koerper. Anders als die alte raute ist er KEIN fester pixelwert --
        er ist `body.radius * camera.scale` und schrumpft mit heraus-zoomen
        auf nichts, genau wie die koerperscheibe selbst.
        """
        if not (math.isfinite(sx) and math.isfinite(sy) and math.isfinite(r_px)):
            return
        if r_px < 0.75:
            return
        if (sx < -r_px or sx > self.width + r_px
                or sy < -r_px or sy > self.height + r_px):
            return
        seg = max(12, min(64, int(r_px)))
        ang = np.linspace(0.0, 2.0 * math.pi, seg + 1)
        ring = np.empty((seg + 1, 2), dtype=np.float64)
        ring[:, 0] = float(sx) + float(r_px) * np.cos(ang)
        ring[:, 1] = float(sy) + float(r_px) * np.sin(ang)
        self._draw_polyline(ring, color=color, width=1.0)

    def _draw_orbit_lines(self, bodies, camera, predictor=None, real_dt=0.0):
        """Wo jeder koerper waehrend des VORHERSAGE-FENSTERS entlanglaeuft.

        Dieselbe zeitspanne wie die schiffslinie, punkt fuer punkt in dem
        plot-frame, den seine EIGENE zeit aufspannt. Damit steht die
        endkappe des koerpers fuer "hier ist er, wenn das schiff am ende
        seiner linie ankommt" -- fallen die beiden endkappen zusammen,
        trifft man. Das ist der ganze zweck, und es funktioniert nur, wenn
        beide linien durch dieselbe transformation gehen.

        Eine feste ellipse waere hier schlicht falsch: ein plot-frame ist
        eine ZEITABHAENGIGE abbildung, eine starr transformierte ellipse
        zeigt die bahn also so, wie sie JETZT gerade laege. Im Erd-rahmen
        kam dabei eine Erdbahn um die Sonne heraus, obwohl die Erde dort im
        ursprung steht.
        """
        self.debug_info['orbit_lines_drawn'] = 0
        if not self.orbit_lines_enabled:
            return

        oset = self._ensure_orbit_line_set()

        points = None
        generation = None
        if predictor is not None:
            try:
                points = predictor.get_points()
            except Exception:
                points = None
            generation = getattr(predictor, '_points_generation', None)

        oset.update(
            bodies, points,
            sim_time=self._frame_time_s, real_dt=real_dt,
            reference_body=self.current_reference_body,
            selected_body=self.selected_body,
            generation=generation,
        )

        frame = self._active_frame()
        origin_body = orbit_lines.frame_origin_body(frame)

        # Der ursprungskoerper bekommt keine linie: er steht in seinem
        # eigenen rahmen still.
        drawable = [e for e in oset.entries()
                    if e.reveal > 0.002 and e.alpha > 0.004
                    and e.track is not None and e.track_t is not None
                    and e.track.shape[0] >= 2
                    and e.body is not origin_body]
        if not drawable:
            return

        # EINE tabelle fuer alle: sie stehen alle auf dem fenster des
        # praediktors, also wird die transformation einmal auf einem
        # knotengitter bestimmt statt je koerper und stichprobe.
        track_t = drawable[0].track_t
        table = orbit_lines.FrameAffineTable(
            frame, float(track_t[0]), float(track_t[-1]),
            knot_angle=float(self.orbit_line_knot_angle))
        if not table.valid:
            return

        # Knotentabellen fuer die faint volllinien: EINE JE KOERPER, ueber
        # SEINE periode. Kein gemeinsames gitter wie bei der spur-tabelle --
        # die perioden liegen um groessenordnungen auseinander (Mond 27 d,
        # ein planet jahre), ein ueber die laengste gespanntes gitter liesse
        # der kurzen periode zu wenige knoten und die ursprungs-interpolation
        # explodiert. Gecacht ueber die frames auf (rahmen, fenster), gebaut
        # nur wenn `_recompute` eine neue `full_track_t` geliefert hat.
        full_enabled = bool(self.orbit_line_full_orbit_enabled)
        full_alpha_mult = float(self.orbit_line_full_alpha_mult)
        full_knot_angle = float(self.orbit_line_full_knot_angle)
        table_cache = self._full_orbit_tables
        # Bei jedem frame-wechsel (R, 1, 2) den ganzen cache verwerfen -- so
        # kann eine wiederverwendete id() des frame-objekts keiner alten
        # tabelle einen falschen treffer geben.
        if frame is not getattr(self, '_full_orbit_tables_frame', None):
            table_cache.clear()
            self._full_orbit_tables_frame = frame
        frame_key = id(frame)
        live_keys = set()

        def _full_table_for(entry):
            ft_t = entry.full_track_t
            # Schluessel ueber die FENSTERGRENZEN, nicht id(ft_t): eine
            # freigegebene array-id kann wiederverwendet werden und gaebe der
            # alten tabelle einen falschen treffer. (rahmen, fenster) bestimmt
            # die affine tabelle vollstaendig -- eine kollision ist harmlos.
            key = (frame_key, round(float(ft_t[0]), 3), round(float(ft_t[-1]), 3))
            live_keys.add(key)
            hit = table_cache.get(key)
            if hit is not None:
                return hit if hit.valid else None
            tab = orbit_lines.FrameAffineTable(
                frame, float(ft_t[0]), float(ft_t[-1]),
                knot_angle=full_knot_angle)
            table_cache[key] = tab
            return tab if tab.valid else None

        camera_frame_xy = self._frame_camera_xy(camera)
        scale = abs(float(camera.scale))
        margin = float(self.prediction_visibility_margin_px)
        view_diag = math.hypot(self.width + 2.0 * margin,
                               self.height + 2.0 * margin)
        min_px = float(self.orbit_line_min_screen_px)
        half_w = self.width * 0.5
        half_h = self.height * 0.5
        cap_px = self.ui_px(self.orbit_line_end_cap_px)
        show_caps = bool(self.orbit_line_end_caps)
        drawn = 0
        any_full = False

        for entry in drawable:
            body = entry.body
            base = getattr(body, 'color', (200, 200, 200))
            try:
                cr = min(1.0, max(0.0, float(base[0]) / 255.0 * 0.85))
                cg = min(1.0, max(0.0, float(base[1]) / 255.0 * 0.85))
                cb = min(1.0, max(0.0, float(base[2]) / 255.0 * 0.85))
            except Exception:
                cr = cg = cb = 0.75

            # Faint volllinie zuerst -- ein ganzer umlauf, HINTER der hellen
            # spur, damit die enthuellte linie oben liegt. Gleiche
            # transformations-pipeline `koerper(t) - ursprung(t)`, also im
            # plot-frame automatisch richtig.
            if (full_enabled and getattr(entry, 'full_track', None) is not None
                    and entry.full_track_t is not None
                    and entry.full_track.shape[0] >= 2):
                fa = float(entry.alpha) * full_alpha_mult
                full_table = _full_table_for(entry) if fa > 0.003 else None
                if full_table is not None:
                    ftrack = entry.full_track
                    # ALLE stichproben projizieren -- kein stride. Der stride
                    # oben schaetzt die zeichen-aufloesung aus `track_len`, und
                    # das ist die WELT-bogenlaenge; ueber eine ganze periode
                    # traegt die eltern-heliozentrik da das zehn- bis
                    # hundertfache der plot-frame-laenge hinein. Es sind ohnehin
                    # nur 0-3 volllinien, `table.project` ist numpy-vektorisiert.
                    ffx, ffy = full_table.project(
                        entry.full_track_t,
                        np.ascontiguousarray(ftrack[:, 0]),
                        np.ascontiguousarray(ftrack[:, 1]))
                    fsx = half_w + (ffx - camera_frame_xy[0]) * scale
                    fsy = half_h - (ffy - camera_frame_xy[1]) * scale
                    self._draw_frame_polyline(
                        fsx, fsy, (cr, cg, cb, fa),
                        float(self.orbit_line_width), min_screen_px=min_px)

            # Enthuellung: die linie rollt sich VOM KOERPER AUS ab.
            total = int(entry.track.shape[0])
            reveal = max(0.0, min(1.0, float(entry.reveal)))
            n_show = max(2, int(math.ceil(reveal * total)))

            # Gezeichnet wird nur so fein, wie die fehlerschranke verlangt.
            # Als kruemmungsradius dient die bogenlaenge selbst -- fuer eine
            # fast gerade kurve ist das eine unterschaetzung, also zu viele
            # punkte statt zu wenige.
            arc_px = entry.track_len * scale * reveal
            r_eff = max(1.0, arc_px / (2.0 * math.pi))
            stride = orbit_lines.polyline_stride(
                n_show, arc_px, r_eff, view_diag,
                float(self.orbit_line_tolerance_px))
            idx = orbit_lines.stride_indices(n_show, stride)

            fx, fy = table.project(
                entry.track_t[idx],
                np.ascontiguousarray(entry.track[idx, 0]),
                np.ascontiguousarray(entry.track[idx, 1]))
            sx = half_w + (fx - camera_frame_xy[0]) * scale
            sy = half_h - (fy - camera_frame_xy[1]) * scale

            if self._draw_frame_polyline(
                    sx, sy, (cr, cg, cb, float(entry.alpha)),
                    float(self.orbit_line_width), min_screen_px=min_px):
                drawn += 1

            # Die endkappe steht fuer den koerper zur ENDZEIT -- solange die
            # linie nicht ganz da ist, endet sie irgendwo dazwischen und
            # duerfte nicht als messpunkt gelesen werden.
            if show_caps and reveal > 0.995:
                any_full = True
                body_r_px = float(getattr(body, 'radius', 0.0) or 0.0) * scale
                self._draw_body_disc_outline(
                    float(sx[-1]), float(sy[-1]), body_r_px,
                    (cr, cg, cb, float(entry.alpha)))

        # Die kappe des schiffs nur, wenn es etwas zu vergleichen gibt --
        # und ueber den weg der GEZEICHNETEN linie, damit sie auf deren ende
        # sitzt und nicht daneben.
        if show_caps and any_full and points is not None and len(points) >= 2:
            try:
                last = points[-1]
                csx, csy = self._world_to_screen_xy_at_time(
                    float(last[0]), float(last[1]), camera, float(last[2]),
                    camera_frame_xy=camera_frame_xy)
                self._draw_end_cap(csx, csy, (1.0, 1.0, 1.0, 0.9), cap_px)
            except Exception:
                pass

        self.debug_info['orbit_lines_drawn'] = drawn

        # Volllinien-tabellen aufraeumen: alles, was dieses bild nicht mehr
        # gebraucht hat (frame gewechselt, neue full_track_t nach recompute).
        if len(table_cache) > len(live_keys):
            for stale in [k for k in table_cache if k not in live_keys]:
                del table_cache[stale]

    def _emit_render_benchmark(self, timings):
        if not self.render_benchmark_debug:
            return
        try:
            self._render_benchmark_frame += 1
            every = max(1, int(self.render_benchmark_every_n_frames))
            if self._render_benchmark_frame % every != 0:
                return
            pred = dict(getattr(self, "_last_prediction_render_stats", {}) or {})
            print(
                "RENDER_BENCH: "
                f"frame_ms={timings.get('frame_ms', 0.0):.3f} "
                f"bodies_ms={timings.get('bodies_ms', 0.0):.3f} "
                f"predictor_prepare_ms={pred.get('prepare_ms', 0.0):.3f} "
                f"predictor_draw_ms={pred.get('draw_ms', 0.0):.3f} "
                f"predictor_raw_in={pred.get('raw_in', 0)} "
                f"scanned={pred.get('scanned', 0)} "
                f"visible={pred.get('visible', 0)} "
                f"drawn={pred.get('drawn', 0)} "
                f"skipped_by_stride={pred.get('skipped_by_stride', 0)} "
                f"clipped_or_rejected={pred.get('clipped_or_rejected', 0)} "
                f"cache_hit={pred.get('cache_hit', False)} "
                f"background_ms={timings.get('background_ms', 0.0):.3f} "
                f"reference_trails_ms={timings.get('reference_trails_ms', 0.0):.3f} "
                f"orbit_lines_ms={timings.get('orbit_lines_ms', 0.0):.3f} "
                f"orbit_lines_drawn={self.debug_info.get('orbit_lines_drawn', 0)} "
                f"hud_ms={timings.get('hud_ms', 0.0):.3f} "
                f"fxaa_ms={timings.get('fxaa_ms', 0.0):.3f} "
                f"overlay_ms={timings.get('overlay_ms', 0.0):.3f} "
                f"swap_or_present_ms={timings.get('swap_or_present_ms', 0.0):.3f}",
                flush=True,
            )
        except Exception:
            pass

    def render(self, bodies, camera, prediction_points=None, predictor=None, sim_time=None, reference_body=None, ship_control=None, real_dt=0.0, selected_body=None):
        frame_t0 = time.perf_counter()
        timings = {
            'bodies_ms': 0.0,
            'background_ms': 0.0,
            'reference_trails_ms': 0.0,
            'orbit_lines_ms': 0.0,
            'hud_ms': 0.0,
            'fxaa_ms': 0.0,
            'overlay_ms': 0.0,
            'swap_or_present_ms': 0.0,
        }

        if sim_time is not None:
            self.set_frame_time(sim_time)
        self.current_reference_body = reference_body
        self.selected_body = selected_body
        self._advance_selection_phases(real_dt)
        self._ship_zoom_factor = self._ship_zoom_shrink_factor(
            getattr(camera, 'scale', None))
        # Fuer alles, was tief im zeichenweg ein echtes zeit-delta braucht
        # (die abgasfahne des schiffs) und keinen eigenen parameter hat.
        self._frame_real_dt = max(0.0, float(real_dt))
        self._dbg_ship_control = ship_control
        try:
            self._current_body_index_by_id = {id(body): idx for idx, body in enumerate(bodies)}
        except Exception:
            self._current_body_index_by_id = {}

        reference_t0 = time.perf_counter()
        self._record_reference_trajectories(bodies)
        timings['reference_trails_ms'] += (time.perf_counter() - reference_t0) * 1000.0

        # Optional periodic debug output to inspect frame transforms.
        if getattr(self, 'debug_frame', False):
            self._frame_debug_counter += 1
            if self._frame_debug_counter % getattr(self, '_frame_debug_period', 30) == 0:
                try:
                    sun = next((b for b in bodies if 'sonn' in getattr(b, 'name', '').lower() or getattr(b, 'name', '').lower() in ('sun', 'sonne')), None)
                    earth = next((b for b in bodies if getattr(b, 'name', '').lower() in ('earth', 'erde')), None)
                    active = self._active_frame()
                    label = getattr(active, 'label', None)
                    if sun is not None and earth is not None:
                        swx, swy = float(sun.position.x), float(sun.position.y)
                        exx, exy = float(earth.position.x), float(earth.position.y)
                        sfx, sfy = self._frame_transform_xy(swx, swy)
                        efx, efy = self._frame_transform_xy(exx, exy)
                        print(f"FRAME_DBG: label={label} time={self._frame_time_s:.3f} sun_world=({swx:.6e},{swy:.6e}) sun_frame=({sfx:.6e},{sfy:.6e}) earth_world=({exx:.6e},{exy:.6e}) earth_frame=({efx:.6e},{efy:.6e})")
                except Exception:
                    pass

        self.debug_info['bodies_rendered'] = 0
        self.debug_info['bodies_culled'] = 0
        self.debug_info['bodies_as_icon'] = 0
        self.debug_info['bodies_vector'] = 0

        # Beleuchtung: EINE quelle je frame, in bildschirmkoordinaten. Jeder
        # koerper leitet daraus nur noch eine richtung ab (`_body_light_dir`).
        if not self.body_vector_style and self._body_style_jobs:
            # Abgeschaltet: laufende bauten interessieren nicht mehr, und
            # liegengelassen wuerden sie das budget dauerhaft belegen.
            self._body_style_jobs.clear()
        self._update_icon_radius_range(bodies)
        self._light_source_body = self._find_light_source(bodies)
        self._light_screen_xy = None
        if self._light_source_body is not None:
            try:
                self._light_screen_xy = self._world_to_screen_xy(
                    float(self._light_source_body.position.x),
                    float(self._light_source_body.position.y),
                    camera,
                    camera_frame_xy=self._frame_camera_xy(camera),
                )
            except Exception:
                self._light_screen_xy = None
        self.debug_info['prediction_points_in'] = 0
        self.debug_info['prediction_points_drawn'] = 0
        self._last_prediction_render_stats = {
            'raw_in': self._points_count(prediction_points),
            'scanned': 0,
            'visible': 0,
            'drawn': 0,
            'skipped_by_stride': 0,
            'clipped_or_rejected': 0,
            'prepare_ms': 0.0,
            'draw_ms': 0.0,
            'cache_hit': False,
        }
        
        # falls FXAA aktiviert ist, rendern nicht-schiff-körper in das FBO und
        # FXAA anwenden. Schiffe werden danach direkt in den haupt-framebuffer
        # gerendert damit predictor (ebenfalls im hauptpuffer gerendert) und
        # das schiff-marker exakt dieselben pixel-koordinaten teilen.
        ship_body = next((b for b in bodies if getattr(b, 'is_ship', False)), None)

        self._deferred_labels.clear()

        if self.enable_fxaa and self.fbo:
            target_fbo = self.fbo
        else:
            target_fbo = self.ctx.screen
        target_fbo.use()
        target_fbo.clear(*self._clear_color)

        # Unterste schicht: sternenfeld + gitter (background.py). Ersetzt
        # faktisch den clear darueber, der aber stehen bleibt, falls die
        # ebene abgeschaltet ist oder ein shader ausgefallen ist.
        background_t0 = time.perf_counter()
        self._draw_background(camera, real_dt)
        timings['background_ms'] = (time.perf_counter() - background_t0) * 1000.0

        reference_t0 = time.perf_counter()
        self._draw_reference_trajectories(bodies, camera)
        timings['reference_trails_ms'] += (time.perf_counter() - reference_t0) * 1000.0

        # Bahnlinien VOR den koerpern: so verdeckt ein koerper seine eigene
        # linie, statt von ihr durchkreuzt zu werden.
        orbit_t0 = time.perf_counter()
        self._draw_orbit_lines(bodies, camera, predictor=predictor, real_dt=real_dt)
        timings['orbit_lines_ms'] = (time.perf_counter() - orbit_t0) * 1000.0

        # Render all non-ship bodies first (they may be FXAA-processed).
        bodies_t0 = time.perf_counter()
        for body in bodies:
            if getattr(body, 'is_ship', False):
                continue

            self._draw_body(body, camera)
        timings['bodies_ms'] += (time.perf_counter() - bodies_t0) * 1000.0

        prediction_has_points = self._points_count(prediction_points) > 0
        prediction_drawn = False

        if prediction_has_points and self.enable_fxaa and self.fbo and not self.prediction_bypass_fxaa:
            self.draw_prediction(prediction_points, camera, predictor=predictor)
            prediction_drawn = True

        if self.enable_fxaa and self.fbo:
            # Zurück zum Standard-Framebuffer and apply FXAA post-process
            self.ctx.screen.use()
            fxaa_t0 = time.perf_counter()
            self._apply_fxaa()
            timings['fxaa_ms'] += (time.perf_counter() - fxaa_t0) * 1000.0

        # Ab hier wird direkt in den haupt-framebuffer gezeichnet (predictor,
        # schiff, HUD). Blending ist global aktiv (ctx.enable in _init_opengl,
        # von _apply_fxaa wiederhergestellt); die alten projektions-resets der
        # fixed-function-pipeline entfallen.
        if prediction_has_points and not prediction_drawn:
            self.draw_prediction(prediction_points, camera, predictor=predictor)

        # Schiff-Marker im Haupt-Framebuffer zeichnen, damit er visuell
        # genau mit dem Prädiktor-Startpunkt übereinstimmt.
        if ship_body is not None:
            bodies_t0 = time.perf_counter()
            # Orientierungs-snap ANWENDEN bevor der pfeil gezeichnet wird, mit
            # demselben frame + _frame_time_s wie die vektoren/der pfeil — so
            # ist die nase exakt an die gezeichneten prograde/normal-vektoren
            # gebunden (keine zeit-/konventionsdrift).
            self._apply_orientation_snap(
                ship_body, ship_control, reference_body, prediction_points, real_dt
            )
            self._draw_body(ship_body, camera)
            self.draw_ship_thrust_vector(ship_body, camera)
            self.draw_ship_orientation_debug_vectors(
                ship_body, camera, reference_body=reference_body,
                prediction_points=prediction_points,
            )
            timings['bodies_ms'] += (time.perf_counter() - bodies_t0) * 1000.0

        # Auswahl-markierung ebenfalls nach dem FXAA-resolve, aus demselben
        # grund wie die beschriftungen -- und vor ihnen, damit ein label nicht
        # unter einem pfeil verschwindet.
        self._draw_selection_marker(camera)

        # Körper-beschriftungen erst jetzt zeichnen -- nach dem FXAA-resolve,
        # damit der kantenfilter den text nicht verschmiert (siehe _draw_body).
        for name, label_x, label_y in self._deferred_labels:
            self._blit_text_topdown(name, label_x, label_y, self.font_small)

        hud_t0 = time.perf_counter()
        self._render_hud(camera, predictor)
        timings['hud_ms'] += (time.perf_counter() - hud_t0) * 1000.0
        # Der buffer-swap liegt NICHT mehr hier, sondern in der hauptschleife
        # (test.py -> present()). Overlays, die zuletzt zeichnen muessen
        # (ImGui-devtools, spaeter das custom-HUD), brauchen die luecke
        # zwischen "welt fertig gezeichnet" und "swap".
        timings['swap_or_present_ms'] = 0.0
        timings['overlay_ms'] = 0.0
        timings['frame_ms'] = (time.perf_counter() - frame_t0) * 1000.0
        self._render_t0 = frame_t0
        # Ende von render(). Bezugspunkt fuer `overlay_ms` -- siehe present().
        self._render_end = time.perf_counter()
        self.last_frame_timings = timings
        self._emit_render_benchmark(timings)

    def present(self):
        """Fuehrt den buffer-swap aus und schreibt die swap-zeit in die timings.

        Von der hauptschleife aufzurufen, NACHDEM alle overlays gezeichnet sind.

        `frame_ms` bleibt dabei stehen: es ist die dauer von render() SELBST.
        Frueher wurde es hier auf "render-start bis nach dem swap" gesetzt,
        und weil `rend_calc` daraus als `frame_ms - swap` gebildet wird, lief
        alles, was zwischen render() und present() gezeichnet wird -- vor
        allem das spieler-HUD (`ui_root.render()`, gemessen ~8 ms median) --
        stillschweigend unter "render calc". Das ist die haelfte der zahl,
        und sie stand an der falschen stelle. Die luecke heisst jetzt
        `overlay_ms` und wird getrennt ausgewiesen.
        """
        swap_t0 = time.perf_counter()
        pygame.display.flip()
        swap_ms = (time.perf_counter() - swap_t0) * 1000.0
        timings = self.last_frame_timings
        if isinstance(timings, dict):
            timings['swap_or_present_ms'] = swap_ms
            end = getattr(self, '_render_end', None)
            if end is not None:
                overlay_ms = (swap_t0 - end) * 1000.0
                # Wer render() ohne present() aufruft (die GL-tests tun das),
                # hinterlaesst einen alten `_render_end` -- dann waere die
                # differenz unsinnig gross oder negativ.
                timings['overlay_ms'] = overlay_ms if overlay_ms >= 0.0 else 0.0

    def draw_ship_thrust_vector(self, ship, camera):
        if ship is None:
            return

        try:
            direction = getattr(ship, "last_thrust_direction", None)
            if direction is None:
                return

            vx = float(direction.x)
            vy = float(direction.y)

            frame = self._active_frame()
            try:
                vx, vy = frame.to_this_frame_vector_xy(self._frame_time_s, vx, vy)
            except Exception:
                pass

            mag = math.hypot(vx, vy)
            if mag <= 1e-12:
                return

            vx /= mag
            vy /= mag

            sx, sy = self._world_to_screen_xy(float(ship.position.x), float(ship.position.y), camera)
            length_px = 45.0
            ex = sx + vx * length_px
            ey = sy - vy * length_px

            self._draw_polyline([(sx, sy), (ex, ey)], color=(1.0, 0.5, 0.1, 0.95), width=2.0)
        except Exception:
            return

    def active_plotting_frame(self):
        """Public accessor for the frame the ship control uses to hold a snap."""
        return self._active_frame()

    def orbital_frame_directions(self, ship, reference_body=None, prediction_points=None):
        """The frame-space orbital directions used to draw the overlay vectors.

        Single source of truth for both the debug vectors and the orientation
        snap: prograde/normal_in are the tangent/inward of the *drawn* predictor
        line in the active plotting frame; retrograde/antinormal are their
        opposites. Evaluated at the renderer's current ``_frame_time_s`` — the
        same instant the ship arrow is drawn.
        """
        frame = self._active_frame()
        if reference_body is None:
            reference_body = getattr(self, "current_reference_body", None)
        ref_pos = getattr(reference_body, "position", None)
        return frame, apparent_orbital_directions(
            frame, self._frame_time_s, ship.position, ship.velocity, ref_pos,
            points=prediction_points,
        )

    def _apply_orientation_snap(self, ship, ship_control, reference_body,
                                prediction_points, real_dt):
        """Tie the ship nose to the drawn orbital vector for the latched snap.

        Computes the world heading whose *drawn* arrow coincides with the
        frame-space snap vector, using ``heading_from_this_frame`` at the SAME
        ``_frame_time_s`` that ``_draw_body`` uses to draw the arrow. This makes
        the nose lock onto the on-screen prograde/normal vector exactly, with no
        dependence on sim_dt or frame rotation rate. The ship's stored ``theta``
        stays in world space (physics remains absolute); only the render-time
        transform is inverted here.
        """
        if ship is None or ship_control is None:
            return
        mode = getattr(ship_control, "snap_mode", None)
        if not mode:
            return
        try:
            frame, directions = self.orbital_frame_directions(
                ship, reference_body, prediction_points
            )
            d = directions.get(mode)
            if d is None:
                return
            # `theta` ist im uhrzeigersinn gemessen (siehe _draw_ship_sprite und
            # schiff.apply_thrust: nasenrichtung = (cos theta, -sin theta)).
            # Damit die nase auf der frame-richtung d landet, muss also
            # (cos theta_f, -sin theta_f) == d gelten -> d.y negiert messen.
            ang_frame = math.atan2(-float(d.y), float(d.x))
            try:
                theta_target = frame.heading_from_this_frame(self._frame_time_s, ang_frame)
            except Exception:
                theta_target = ang_frame
            ship_control.orient_towards_angle(theta_target, real_dt)
        except Exception:
            return

    def draw_ship_orientation_debug_vectors(self, ship, camera, reference_body=None,
                                            prediction_points=None):
        """Debug overlay: always draws prograde (green) + normal-inward (magenta).

        Directions come from ``apparent_orbital_directions`` fed the actual
        predictor polyline (``prediction_points``) — i.e. the tangent of the
        drawn line as it appears in the active plotting frame — so they already
        live in frame space and only need the screen y-flip, exactly like the
        drawn trajectory. This keeps them glued to the predictor line as it
        changes shape, including in rotating/translating frames.
        """
        if ship is None:
            return

        try:
            frame, directions = self.orbital_frame_directions(
                ship, reference_body, prediction_points
            )

            sx, sy = self._world_to_screen_xy(float(ship.position.x), float(ship.position.y), camera)
            length_px = 55.0

            for key, color in (
                ("prograde", (0.2, 1.0, 0.35, 0.95)),
                ("normal_in", (0.9, 0.3, 1.0, 0.95)),
            ):
                d = directions.get(key)
                if d is None:
                    continue
                ex = sx + float(d.x) * length_px
                ey = sy - float(d.y) * length_px
                # Stash the ACTUAL drawn pixel direction of each vector so the
                # diagnostic can compare raw screen geometry (non-derived).
                if key == "normal_in":
                    self._last_normal_screen_dir = (ex - sx, ey - sy)
                elif key == "prograde":
                    self._last_prograde_screen_dir = (ex - sx, ey - sy)
                self._draw_polyline([(sx, sy), (ex, ey)], color=color, width=2.0)

            self._debug_orientation_angles(ship, camera, frame, directions, sx, sy)
        except Exception:
            return

    def _debug_orientation_angles(self, ship, camera, frame, directions, sx, sy):
        """Env-guarded (SPACESIM_DEBUG_ORIENT=1) screen-space angle report.

        Prints, in one common screen convention, the heading of: the actual
        predictor orbit line, my green prograde, the blue velocity vector, and
        the ship nose. Whichever one disagrees is the culprit for the reported
        45 deg offset. Behavior-neutral: only prints, throttled.
        """
        if os.environ.get("SPACESIM_DEBUG_ORIENT", "0").strip().lower() in ("0", "", "false", "off", "no"):
            return
        self._debug_orient_counter = getattr(self, "_debug_orient_counter", 0) + 1
        if self._debug_orient_counter % 30 != 1:
            return

        def sdeg(dx, dy):
            # Screen-space heading: vectors are drawn as (dx, -dy), so the
            # on-screen angle of a frame-space direction is atan2(-dy, dx).
            return math.degrees(math.atan2(-dy, dx))

        parts = []

        # RAW drawn pixel angles (from the actual vertices, y-down screen space).
        def rawdeg(v):
            if v is None:
                return None
            return math.degrees(math.atan2(v[1], v[0]))

        norm_v = getattr(self, "_last_normal_screen_dir", None)
        arrow_v = getattr(self, "_last_arrow_screen_dir", None)
        norm_deg = rawdeg(norm_v)
        # DISPLAYED arrow angle: the arrow renders under gluOrtho2D bottom-up
        # while vectors render via the line shader top-down, so the arrow's
        # on-screen y is the negation of its input y.
        arrow_deg = None if arrow_v is None else math.degrees(math.atan2(-arrow_v[1], arrow_v[0]))

        if norm_deg is not None:
            parts.append(f"magenta_raw={norm_deg:8.3f}")
        if arrow_deg is not None:
            parts.append(f"arrow_raw={arrow_deg:8.3f}")

        # Per-sample rotation of each, so co-rotation vs counter-rotation is
        # directly visible (this is the user's actual complaint).
        prev = getattr(self, "_dbg_prev_raw", None)
        if prev is not None and norm_deg is not None and arrow_deg is not None:
            dmag = (norm_deg - prev[0] + 180.0) % 360.0 - 180.0
            darr = (arrow_deg - prev[1] + 180.0) % 360.0 - 180.0
            sense = "SAME" if (dmag * darr) >= 0 else "OPPOSITE"
            parts.append(f"d_mag={dmag:+7.3f} d_arrow={darr:+7.3f} [{sense}]")
        if norm_deg is not None and arrow_deg is not None:
            self._dbg_prev_raw = (norm_deg, arrow_deg)
            gap = (arrow_deg - norm_deg + 180.0) % 360.0 - 180.0
            parts.append(f"gap={gap:+7.3f}")

        mode = getattr(getattr(self, "_dbg_ship_control", None), "snap_mode", None)
        frame_label = getattr(frame, "label", frame.__class__.__name__)
        print(f"ORIENT_DBG: snap={mode} " + "  ".join(parts) + f"  frame='{frame_label}'")

    def _ship_relative_speed_m_s(self, ship, reference_body=None):
        if ship is None:
            return None

        try:
            vx = float(ship.velocity.x)
            vy = float(ship.velocity.y)
        except Exception:
            return None

        if reference_body is not None:
            try:
                vx -= float(reference_body.velocity.x)
                vy -= float(reference_body.velocity.y)
            except Exception:
                pass

        return math.hypot(vx, vy)

    def _ship_frame_speed_m_s(self, ship, dt_s=1.0):
        """
        Returns the ship's apparent speed in the active plotting frame.

        This respects translated, rotating, target-overlay, and time-dependent
        frames by finite-differencing the active frame transform. It does not
        use the clamped visual velocity vector length.
        """
        if ship is None:
            return None

        try:
            t0 = float(self._frame_time_s)
            dt = max(1e-3, float(dt_s))

            x0 = float(ship.position.x)
            y0 = float(ship.position.y)
            vx = float(ship.velocity.x)
            vy = float(ship.velocity.y)

            frame = self._active_frame()

            fx0, fy0 = frame.to_this_frame_xy(t0, x0, y0)
            fx1, fy1 = frame.to_this_frame_xy(
                t0 + dt,
                x0 + vx * dt,
                y0 + vy * dt,
            )

            dvx = float(fx1) - float(fx0)
            dvy = float(fy1) - float(fy0)

            return math.hypot(dvx, dvy) / dt
        except Exception:
            return None

    def _format_speed_label(self, speed_m_s):
        if speed_m_s is None:
            return ""

        speed_m_s = float(speed_m_s)
        if speed_m_s >= 1000.0:
            return f"{speed_m_s / 1000.0:.2f} km/s"

        return f"{speed_m_s:.1f} m/s"
    
    def _find_light_source(self, bodies):
        """Der koerper, der das system beleuchtet.

        `light_intensity > 0` gewinnt; sonst der massereichste koerper. Der
        fallback ist absicht: ein selbst gebautes system ohne das feld soll
        trotzdem beleuchtet aussehen, und der schwerste koerper ist dort
        praktisch immer der stern.
        """
        best = None
        best_mass = -1.0
        for candidate in bodies:
            if getattr(candidate, 'is_ship', False):
                continue
            if float(getattr(candidate, 'light_intensity', 0.0)) > 0.0:
                return candidate
            mass = float(getattr(candidate, 'mass', 0.0))
            if mass > best_mass:
                best = candidate
                best_mass = mass
        return best

    def _draw_body(self, body, camera):
        camera_frame_xy = self._frame_camera_xy(camera)
        x, y = self._world_to_screen_xy(
            float(body.position.x),
            float(body.position.y),
            camera,
            camera_frame_xy=camera_frame_xy,
        )
        screen_pos = (x, y)
        r, g, b = body.color[0] / 255.0, body.color[1] / 255.0, body.color[2] / 255.0
        x, y = float(screen_pos[0]), float(screen_pos[1])

        if body.is_ship:
            # Schiff: feste bildschirmgröße (pfeil), nie gecullt, nie als icon.
            self.debug_info['bodies_rendered'] += 1
            theta_frame = float(getattr(body, 'theta', 0.0))
            try:
                theta_frame = self._active_frame().transform_heading(self._frame_time_s, theta_frame)
            except Exception:
                pass
            self._draw_ship_sprite(body, x, y, r, g, b, theta_override=theta_frame)
            # Das Schiff traegt KEINEN schwebenden text mehr -- name und
            # geschwindigkeit standen frueher fest ueber/unter der silhouette.
            # Beide leben im spieler-HUD (navball-cluster); der name erscheint
            # ueber das auswahl-label, wenn das schiff angeklickt wird.
            return

        # --- Nicht-Schiff-Körper: off-screen-cull + größen-schwelle (icon-swap) ---
        # Echter, UNgeklemmter bildschirmradius. Statt den körper (alt) auf
        # min. 3px zu klemmen und dauerhaft als winzige scheibe zu zeichnen,
        # lassen wir ihn unter die schwelle schrumpfen und tauschen ihn dann
        # nahtlos gegen ein positions-icon konstanter größe.
        icon_min_radius_px = float(self.body_icon_min_radius_px)
        true_radius_px = float(body.radius) * float(camera.scale)
        as_icon = true_radius_px < icon_min_radius_px

        # Off-screen-cull (NUR rendering, physik unberührt): die marge deckt für
        # sichtbare körper den glow (~2.5x radius) ab, damit randständige große
        # körper nicht fälschlich verschwinden. Vollständig off-screen-körper
        # werden gar nicht erst gezeichnet (kein shader-/icon-aufruf).
        cull_margin_px = (icon_min_radius_px if as_icon else true_radius_px * 2.5) + 8.0
        if not self._is_on_screen(x, y, cull_margin_px):
            self.debug_info['bodies_culled'] = self.debug_info.get('bodies_culled', 0) + 1
            return

        self.debug_info['bodies_rendered'] += 1

        if as_icon:
            # Körper komplett de-rendern; nur die positions-marke zeichnen.
            # Die groesse haengt am PHYSISCHEN radius, nicht am (hier winzigen
            # bis nahe-null) bildschirmradius -- siehe `_body_icon_draw_radius_px`.
            self.debug_info['bodies_as_icon'] = self.debug_info.get('bodies_as_icon', 0) + 1
            icon_draw_radius_px = self._body_icon_draw_radius_px(float(body.radius))
            self._draw_body_icon(body, x, y, icon_draw_radius_px, r, g, b, 1.0)
            # Der name haengt NICHT an der zeichengroesse: ein angewaehlter
            # mond soll auch als marke lesbar beschriftet sein. Der zoom-modus
            # dagegen misst den ECHTEN radius, nicht die marke.
            self._queue_body_label(body, x, y, icon_draw_radius_px, screen_pos,
                                   size_radius_px=true_radius_px)
            return

        # --- Voller körper (disc + glow + atmosphäre) bei echter größe ---
        # Gleitkomma-Radius für Label-Anker beibehalten, um 1-Pixel-Flackern beim
        # Zoomen zu vermeiden. radius_px >= icon_radius_px ist hier garantiert.
        radius_px = true_radius_px
        radius = max(3, int(round(radius_px)))  # integer radius for geometry

        if hasattr(body, 'atmosphere_color'):
            r1, g1, b1 = body.atmosphere_color[0] / 255.0, body.atmosphere_color[1] / 255.0, body.atmosphere_color[2] / 255.0
        else:
            r1, g1, b1 = r, g, b

        has_atmos = bool(getattr(body, 'has_atmosphere', False))
        atmos_density = float(getattr(body, 'atmos_density', 0.0)) if has_atmos else 0.0
        light_intensity = float(getattr(body, 'light_intensity', 0.0))

        # Lichtrichtung und detailgrad bestimmen, BEVOR die scheibe gezeichnet
        # wird: `fade` verdunkelt die scheibe genau so weit, wie die vektor-
        # zeichnung darueber sie ersetzt.
        light, emissive = self._body_light_dir(body, x, y)
        fade = self._body_detail_fade(radius_px)
        style_layers = []
        if fade > 0.0:
            for detail, weight in self._body_detail_levels(radius_px):
                entry = self._body_style_entry(body, detail)
                if entry is not None:
                    style_layers.append((entry, weight))
        if not style_layers:
            # Noch nicht gebaut (budget) oder abgeschaltet: die alte flache
            # scheibe bleibt stehen, statt einen leeren dunklen kreis zu zeigen.
            fade = 0.0
        else:
            # Waehrend einer ueberblendung fehlt die zweite stufe vielleicht
            # noch. Dann traegt die vorhandene das volle bild, statt dass die
            # zeichnung fuer einen frame halb durchsichtig wird.
            total = sum(weight for _entry, weight in style_layers)
            if total > 1e-6:
                style_layers = [(entry, weight / total)
                                for entry, weight in style_layers]

        # GLSL-Shader zeichnet Scheibe + Glow + Atmosphäre in einem Quad.
        # (Kein immediate-mode-fallback mehr: ohne body-shader wird der körper
        # nicht gezeichnet, der fehler steht in debug_info['shader_error'].)
        self._draw_body_glsl(
            x,
            y,
            radius_px,
            (r, g, b),
            (r1, g1, b1),
            atmos_density,
            light_intensity,
            light=light,
            emissive=emissive,
            surface_mix=fade,
            glow=float(self.body_glow_alpha) * fade,
        )

        drawn = False
        for entry, weight in style_layers:
            if self._draw_body_vector(entry, x, y, radius_px,
                                      light, emissive, fade * weight):
                drawn = True
        if drawn:
            self.debug_info['bodies_vector'] = (
                self.debug_info.get('bodies_vector', 0) + 1)

        # --- Ueberblendung marke -> koerper -------------------------------
        # Knapp ueber der schwelle ist der koerper zwar schon "echt", sieht
        # aber noch nicht danach aus: eine 8-px-scheibe mit limbus ist etwas
        # anderes als ein zellmuster, und ein harter tausch bei exakt gleichem
        # radius poppt trotzdem. Der koerper ist oben also ganz normal
        # gezeichnet, und die marke wird DARUEBER ausgeblendet -- eine echte
        # ueberblendung, ohne dass der koerper-zeichenweg davon etwas wissen
        # muss.
        icon_fade = self._body_icon_fade(true_radius_px)
        if icon_fade > 0.0:
            icon_draw_radius_px = self._body_icon_draw_radius_px(float(body.radius))
            self._draw_body_icon(body, x, y, icon_draw_radius_px, r, g, b, icon_fade)

        self._queue_body_label(body, x, y, radius_px, screen_pos)

    def _wants_body_label(self, body, radius_px):
        """Ob der name dieses koerpers gerade angeschrieben wird.

        `body_label_mode` entscheidet, WAS die beschriftung ausloest --
        `"selected"` die auswahl, `"zoom"` der bildschirmradius (das alte
        verhalten), `"both"` beides. Der auswahl-fall haengt bewusst NICHT
        an der groesse: sonst haette gerade der weit entfernte koerper, den
        man anklickt, um ihn zu finden, keinen namen.
        """
        mode = str(getattr(self, 'body_label_mode', 'selected')).strip().lower()
        selected = (body is not None and body is self.selected_body)
        try:
            big = float(radius_px) > float(self.body_label_min_radius_px)
        except (TypeError, ValueError):
            big = False
        if mode == 'zoom':
            return big
        if mode == 'both':
            return selected or big
        return selected

    def _queue_body_label(self, body, lx, ly, radius_px, screen_pos=None,
                          size_radius_px=None):
        """Den namen eines koerpers fuer die zeichnung NACH dem FXAA vormerken.

        NICHT sofort zeichnen: koerper laufen in den FXAA-FBO, und FXAA ist
        ein kantenfilter -- ueber gerastertem text macht er aus 34.7 % voll
        deckenden pixeln 5.3 % und verschmiert die glyphen ueber 55 % mehr
        pixel. Die beschriftung wird deshalb gesammelt und in render() NACH
        dem FXAA-resolve gezeichnet, so wie schiff und apsis-marker es schon
        immer wurden.

        `lx, ly` ist die FRAME-AWARE bildschirmposition aus
        `_world_to_screen_xy`, nicht `camera.world_to_screen`: in rotierenden
        plot-frames weichen beide voneinander ab und das label loest sich vom
        koerper. `radius_px` ist der bezugsradius, an dem der text haengt --
        der echte bildschirmradius beim vollen koerper, die marken-groesse bei
        der marke.

        `size_radius_px` ist davon getrennt: es ist die GROESSE, nach der
        `body_label_mode = "zoom"` entscheidet, und das ist immer der echte
        bildschirmradius des koerpers. Beides zu vermengen war lange folgenlos,
        weil die marke mit 4 px unter `body_label_min_radius_px` (5) lag --
        mit 8 px lag sie darueber, und ploetzlich trug im zoom-modus jeder
        winzige mond seinen namen. Der anker haengt an der ZEICHNUNG, die
        entscheidung am KOERPER.
        """
        if size_radius_px is None:
            size_radius_px = radius_px
        if not self._wants_body_label(body, size_radius_px):
            return
        try:
            # Bei ausgewaehltem koerper steht ueber ihm der obere
            # auswahl-pfeil -- `lift` hebt den text darueber hinweg.
            lift = self.selection_label_lift_px(body)
            entry = self._get_label_texture(body.name, self.font_small)
            if entry:
                _, w, h = entry
                label_x = float(lx) - (float(w) / 2.0)
                # ueber den koerper setzen: top-down ist "oben" kleineres y
                label_y = float(ly) - float(radius_px) - lift - 6.0 - float(h)
                self._deferred_labels.append((body.name, label_x, label_y))
            else:
                self._deferred_labels.append((body.name,
                                              float(lx) + float(radius_px)
                                              + lift + 2.0,
                                              float(ly) - 8.0))
        except Exception:
            try:
                self._draw_body_label(
                    body.name,
                    screen_pos if screen_pos is not None else (lx, ly),
                    radius_px)
            except Exception:
                pass

    def _ship_zoom_shrink_factor(self, camera_scale):
        """Massstabs-faktor des schiffs fuer die aktuelle zoomstufe.

        1.0 bei `ship_zoom_shrink_start_scale` und darueber,
        `ship_zoom_shrink_min` bei `ship_zoom_shrink_end_scale` und darunter,
        dazwischen ein smoothstep im LOG-raum der skala. Log, weil zoom
        multiplikativ ist (`camera._ease_scale` interpoliert aus demselben
        grund logarithmisch): linear in `scale` gerechnet waere die ganze
        ueberblendung in der obersten dekade verbraucht und der rest ein
        sprung. Smoothstep statt gerade, damit auch die ENDEN der rampe
        knickfrei sind -- ein linearer verlauf springt am start- und
        endpunkt sichtbar in der aenderungsrate.

        Reine rechnung, kein GL -- damit sie ohne kontext pruefbar ist.
        """
        if not bool(getattr(self, 'ship_zoom_shrink_enabled', True)):
            return 1.0
        try:
            scale = float(camera_scale)
            start = float(self.ship_zoom_shrink_start_scale)
            end = float(self.ship_zoom_shrink_end_scale)
            floor = float(self.ship_zoom_shrink_min)
        except (TypeError, ValueError):
            return 1.0
        floor = max(0.05, min(1.0, floor))
        if not (math.isfinite(scale) and scale > 0.0):
            return 1.0
        if not (start > 0.0 and end > 0.0 and end < start):
            # Unbrauchbar konfiguriert (vertauscht oder gleich): lieber die
            # alte feste groesse als eine division durch null.
            return 1.0
        if scale >= start:
            return 1.0
        if scale <= end:
            return floor
        t = math.log(start / scale) / math.log(start / end)
        t = t * t * (3.0 - 2.0 * t)
        return 1.0 + (floor - 1.0) * t

    def _ship_length_px(self):
        """Gezeichnete schiffslaenge in echten bildschirm-pixeln.

        Basislaenge (design-einheiten -> `ui_px`) x spieler-regler
        `ship_render_scale` x zoom-schrumpfung. EIN weg fuer alle
        zeichenpfade, damit grafik, pfeil-fallback und label-abstand nicht
        auseinanderlaufen.
        """
        return (self.ui_px(self.ship_length_px)
                * max(0.01, float(self.ship_render_scale))
                * max(0.05, float(getattr(self, '_ship_zoom_factor', 1.0))))

    def _ship_half_height_px(self):
        """Halbe hoehe der gezeichneten schiffs-grafik in bildschirm-pixeln.

        Bezugsgroesse fuer alles, was NEBEN dem schiff sitzt (labels). Faellt
        auf die halbe breite des alten pfeils zurueck, wenn die grafik aus ist.
        """
        geo = self._ship_geometry() if self.ship_sprite_enabled else None
        if geo is None:
            return 7.0 * max(0.05, float(getattr(self, '_ship_zoom_factor', 1.0)))
        return self._ship_length_px() * 0.5 * geo.height / geo.length

    def _ship_geometry(self):
        """Die gebaute schiffs-grafik, gecacht bis die akzentfarbe wechselt."""
        cache = self._ship_geometry_cache
        if cache is not None and cache.accent == self.ship_accent_color:
            return cache
        try:
            cache = ship_art.build(self.ship_accent_color)
        except Exception as exc:
            print(f"RENDERER WARNING: schiffs-grafik konnte nicht gebaut werden ({exc})")
            self.ship_sprite_enabled = False
            return None
        self._ship_geometry_cache = cache
        return cache

    def _ship_plume_intensity(self, body, real_dt):
        """Helligkeit der abgasfahne, weich zwischen leerlauf und schub.

        `body.last_thrust_direction` wird in test.py je frame geleert und von
        `schiffcontrol` gesetzt, sobald schub anliegt -- es ist also ein
        echtes "brennt gerade"-signal. Nur schub NACH VORN zuendet die
        hauptduese: beim rueckwaerts-schub (pfeil ab) sitzen die duesen an
        der nase, hinten glimmt dann nur der leerlauf.
        """
        idle = max(0.0, min(1.0, float(self.ship_plume_idle)))
        target = idle
        thrust = getattr(body, 'last_thrust_direction', None)
        if thrust is not None:
            try:
                # Der vergleich laeuft in WELTkoordinaten: theta und der
                # schubvektor sind beide absolut, die frame-transformierte
                # zeichenrichtung waere hier der falsche massstab.
                theta_world = float(getattr(body, 'theta', 0.0))
                dot = (float(thrust.x) * math.cos(theta_world)
                       - float(thrust.y) * math.sin(theta_world))
                if dot > 0.0:
                    target = 1.0
            except Exception:
                target = 1.0
        # Zeitkonstante ~80 ms, mit dem ECHTEN frame-delta gerechnet, damit
        # das aufflammen bei 30 wie bei 240 fps gleich schnell ist.
        dt = max(0.0, float(real_dt))
        k = 1.0 if dt <= 0.0 else min(1.0, dt / 0.08)
        self._ship_plume_level += (target - self._ship_plume_level) * k
        return self._ship_plume_level

    def _draw_ship_sprite(self, body, x, y, r, g, b, theta_override=None):
        """Das schiff aus `ship_art` zeichnen -- in festen bildschirm-pixeln.

        Die grafik liegt im lokalen schiffsraum vor (+x = nase, +y nach oben,
        einheit "SVG-pixel"). Hier wird sie einmal je frame gedreht, auf die
        gewuenschte bildschirmlaenge skaliert und an die schiffsposition
        geschoben; die batches aus `ship_art` sind nur slices in dieses eine
        transformierte array.
        """
        geo = self._ship_geometry() if self.ship_sprite_enabled else None
        if geo is None:
            self._draw_ship_arrow(body, x, y, r, g, b, theta_override=theta_override)
            return

        theta = float(theta_override) if theta_override is not None else float(getattr(body, 'theta', 0.0))

        # Die grafik laeuft ueber die ORTHO-pipeline (y nach oben), die
        # uebergebene position kommt aber aus _world_to_screen_xy (top-down).
        # Ohne diese umrechnung landet das schiff an der ueber die
        # bildschirmmitte gespiegelten stelle -- exakt mittig faellt das nicht
        # auf, abseits der mitte steht es weit neben seiner bahn.
        y = self._ortho_y(y)

        # `theta` ist im UHRZEIGERSINN gemessen: schiff.apply_thrust schiebt
        # entlang Vec2(cos theta, -sin theta), das ist die weltrichtung der
        # nase. Die grafik muss also ebenfalls (cos, -sin) zeigen.
        hx = math.cos(theta)
        hy = -math.sin(theta)
        # Stash the ACTUAL drawn nose screen-direction so diagnostics can compare
        # the real ship pixels against the drawn vectors (non-circular check).
        self._last_arrow_screen_dir = (hx, hy)

        scale = self._ship_length_px() / geo.length

        # Eine drehmatrix fuer das GANZE array: (x', y') = (hx*x - hy*y,
        # hy*x + hx*y). Rechtshaendig, y zeigt in der ortho-konvention nach
        # oben -- die grafik wird also nicht gespiegelt.
        rot = np.array(((hx, hy), (-hy, hx)), dtype=np.float64)
        pts = geo.verts @ rot
        pts *= scale
        pts[:, 0] += x
        pts[:, 1] += y

        def draw(ops, alpha_gain):
            for mode, rgba, width, start, count in ops:
                alpha = float(rgba[3]) * alpha_gain
                if alpha <= 0.002:
                    continue
                # Die koerperfarbe des schiffs wirkt als tint: bei dem weissen
                # standard-schiff ist das die identitaet, ein eingefaerbtes
                # schiff behaelt aber seine kennfarbe.
                color = (rgba[0] * r, rgba[1] * g, rgba[2] * b, alpha)
                if mode == 'lines':
                    self._draw_ortho_shape(
                        pts[start:start + count], color, moderngl.LINES,
                        width=min(4.0, max(1.0, width * scale)),
                    )
                else:
                    self._draw_ortho_shape(
                        pts[start:start + count], color, moderngl.TRIANGLES,
                    )

        plume = self._ship_plume_intensity(body, getattr(self, '_frame_real_dt', 0.0))
        if plume > 0.0:
            draw(geo.plume_ops, plume)
        draw(geo.ops, 1.0)

        if self.debug_predictor:
            # cyan cross = uebergebene screen-position (= der ursprung der grafik)
            size = 3.0
            self._draw_ortho_shape(
                [(x - size, y), (x + size, y), (x, y - size), (x, y + size)],
                color=(0.0, 1.0, 1.0, 1.0),
                mode=moderngl.LINES,
            )

    def _draw_ship_arrow(self, body, x, y, r, g, b, theta_override=None):
        """Der alte dreiecks-pfeil.

        Rueckfallweg, wenn `ship_sprite_enabled` aus ist oder `ship_art` sich
        nicht bauen liess -- bis auf die grafik identisch zu
        `_draw_ship_sprite` (gleiche pixel-groesse, gleiche nasenrichtung).
        """
        # in bildschirm-pixeln zeichnen, damit die schiffgröße nicht mit der
        # welt-geometrie skaliert. Die zoom-schrumpfung (siehe
        # _ship_zoom_shrink_factor) gilt hier genauso wie fuer die grafik --
        # sonst waere der fallback weit herausgezoomt ploetzlich der groessere
        # von beiden.
        zoom = max(0.05, float(getattr(self, '_ship_zoom_factor', 1.0)))
        arrow_length = 18.0 * zoom
        arrow_half_width = 7.0 * zoom
        tail_offset = 6.0 * zoom

        theta = float(theta_override) if theta_override is not None else float(getattr(body, 'theta', 0.0))

        # Der pfeil laeuft ueber die ORTHO-pipeline (y nach oben), die
        # uebergebene position kommt aber aus _world_to_screen_xy (top-down).
        # Ohne diese umrechnung wird der pfeil an der ueber die bildschirmmitte
        # gespiegelten stelle gezeichnet: exakt mittig faellt das nicht auf,
        # abseits der mitte steht das schiff weit neben seiner bahn.
        y = self._ortho_y(y)

        # `theta` ist im UHRZEIGERSINN gemessen: schiff.apply_thrust schiebt
        # entlang Vec2(cos theta, -sin theta), das ist die weltrichtung der
        # nase. Der pfeil muss also ebenfalls (cos, -sin) zeigen.
        # Die positions-korrektur oben aendert daran nichts -- eine
        # verschiebung dreht keine richtung.
        hx = math.cos(theta)
        hy = -math.sin(theta)
        nx = -hy
        ny = hx
        # Stash the ACTUAL drawn nose screen-direction so diagnostics can compare
        # the real arrow pixels against the drawn vectors (non-circular check).
        self._last_arrow_screen_dir = (hx, hy)

        # ursprung anpassen damit der dreiecks-schwerpunkt an (x, y) liegt.
        # der schwerpunkt des dreiecks aus nase und schwanz-ecken liegt
        # entlang der richtung versetzt um (arrow_length - 2*tail_offset)/3
        # in bildschirm-pixeln. verschiebe den lokalen ursprung zurück um diesen
        # betrag damit die welt-position des schiffs dem visuellen mittelpunkt des pfeils entspricht.
        centroid_offset = (arrow_length - 2.0 * tail_offset) / 3.0
        origin_x = x - hx * centroid_offset
        origin_y = y - hy * centroid_offset

        nose_x = origin_x + hx * arrow_length
        nose_y = origin_y + hy * arrow_length
        tail_x = origin_x - hx * tail_offset
        tail_y = origin_y - hy * tail_offset

        left_x = tail_x + nx * arrow_half_width
        left_y = tail_y + ny * arrow_half_width
        right_x = tail_x - nx * arrow_half_width
        right_y = tail_y - ny * arrow_half_width

        self._draw_ortho_shape(
            [(nose_x, nose_y), (left_x, left_y), (right_x, right_y)],
            color=(r, g, b, 1.0),
            mode=moderngl.TRIANGLES,
        )
        # debug: kleine marker zeichnen und einzeilige info ausgeben die
        # den dreiecks-schwerpunkt mit der übergebenen screen-position vergleicht.
        try:
            if self.debug_predictor:
                centroid_x = (nose_x + left_x + right_x) / 3.0
                centroid_y = (nose_y + left_y + right_y) / 3.0
                print(f"PRED_DBG_DRAW: centroid=({centroid_x:.6f},{centroid_y:.6f}) screen_pos=({x:.6f},{y:.6f})")
                # magenta cross = centroid, cyan cross = passed screen pos
                size = 3.0
                self._draw_ortho_shape(
                    [(centroid_x - size, centroid_y), (centroid_x + size, centroid_y),
                     (centroid_x, centroid_y - size), (centroid_x, centroid_y + size)],
                    color=(1.0, 0.0, 1.0, 1.0),
                    mode=moderngl.LINES,
                )
                self._draw_ortho_shape(
                    [(x - size, y), (x + size, y),
                     (x, y - size), (x, y + size)],
                    color=(0.0, 1.0, 1.0, 1.0),
                    mode=moderngl.LINES,
                )
        except Exception:
            pass

    def _points_count(self, points):
        if points is None:
            return 0
        try:
            return len(points)
        except Exception:
            return 0

    def _point_xy(self, point):
        if hasattr(point, 'x') and hasattr(point, 'y'):
            return float(point.x), float(point.y)
        return float(point[0]), float(point[1])

    # ------------------------------------------------------------------
    # Auswahl: anklicken und markieren
    # ------------------------------------------------------------------

    def _pick_radius_px(self, body, camera):
        """Greifradius eines koerpers in bildschirm-pixeln.

        Deckungsgleich mit dem, was `_draw_body` zeichnet: der echte radius,
        nach unten auf die icon-groesse geklemmt (darunter IST der koerper das
        icon). Das schiff ist ein pfeil fester bildschirmgroesse und bekommt
        deshalb einen festen wert.
        """
        if getattr(body, 'is_ship', False):
            return 12.0
        true_radius_px = float(getattr(body, 'radius', 0.0)) * float(camera.scale)
        # Dieselbe funktion wie beim zeichnen: das klickziel deckt sich mit
        # dem, was tatsaechlich zu sehen ist, auch wenn body_icon_size_influence
        # die marke groesser als body_icon_min_radius_px zeichnet.
        icon_radius_px = self._body_icon_draw_radius_px(
            float(getattr(body, 'radius', 0.0)))
        return max(true_radius_px, icon_radius_px)

    def pick_body(self, screen_pos, bodies, camera):
        """Index des koerpers unter `screen_pos` (top-down pixel), sonst None.

        Rechnet ueber DENSELBEN pfad wie das zeichnen (`_world_to_screen_xy`
        mit `_frame_camera_xy`), damit die trefferflaeche im rotierenden wie
        im nicht-rotierenden rahmen genau dort liegt, wo der koerper zu sehen
        ist. Eine eigene, "einfachere" rechnung ueber `camera.world_to_screen`
        waere in jedem bewegten plot-frame daneben.

        Laeuft NUR beim klick, nicht je frame: 28 transformationen.
        """
        if not bodies:
            return None
        try:
            cx = float(screen_pos[0])
            cy = float(screen_pos[1])
        except Exception:
            return None

        camera_frame_xy = self._frame_camera_xy(camera)
        margin = self.ui_px(self.selection_pick_margin_px)

        best_index = None
        best_distance = 0.0
        best_radius = 0.0
        for index, body in enumerate(bodies):
            try:
                sx, sy = self._world_to_screen_xy(
                    float(body.position.x), float(body.position.y),
                    camera, camera_frame_xy=camera_frame_xy,
                )
            except Exception:
                continue
            if not (math.isfinite(sx) and math.isfinite(sy)):
                continue
            radius = self._pick_radius_px(body, camera)
            grab = radius + margin
            dx = sx - cx
            dy = sy - cy
            distance = math.hypot(dx, dy)
            if distance > grab:
                continue
            # Naechster MITTELPUNKT gewinnt, nicht der groesste treffer: sonst
            # verschluckt eine bildfuellende Sonne jeden mond, der als icon
            # davor steht. Bei gleichem abstand der kleinere koerper -- das
            # ist der spezifischere treffer.
            if (best_index is None
                    or distance < best_distance
                    or (distance == best_distance and radius < best_radius)):
                best_index = index
                best_distance = distance
                best_radius = radius
        return best_index

    def selection_label_lift_px(self, body):
        """Wieviel die beschriftung eines koerpers hoeher sitzen muss.

        Der obere pfeil steht genau dort, wo `_draw_body` sonst das label
        anheftet -- sichtbar als text mit einem dreieck darin. Bewusst OHNE
        den puls gerechnet, mit fester zugabe: eine mitatmende beschriftung
        waere unruhiger als die ueberdeckung, die sie behebt.
        """
        if body is not self.selected_body or not self.selection_marker_enabled:
            return 0.0
        span = self.ui_px(float(self.selection_gap_px)
                          + float(self.selection_arrow_length_px))
        return span * (1.0 + 2.0 * float(self.selection_pulse_amount)) + 4.0

    def _advance_selection_phases(self, real_dt):
        """Dreh- und pulsphase der markierung fortschreiben.

        Ueber das ECHTE frame-delta, nicht um einen festen betrag je frame:
        sonst haengt die drehzahl an der bildrate.
        """
        dt = max(0.0, float(real_dt))
        two_pi = 2.0 * math.pi
        spin = math.radians(float(self.selection_spin_deg_per_s)) * dt
        self._selection_spin_phase = (self._selection_spin_phase + spin) % two_pi
        period = max(float(self.selection_pulse_period_s), 1e-3)
        self._selection_pulse_phase = (
            (self._selection_pulse_phase + two_pi * dt / period) % two_pi
        )

    def _selection_marker_vertices(self, cx, cy, body_radius_px):
        """Die 12 ortho-eckpunkte der vier pfeile (4 dreiecke).

        Gibt None zurueck, wenn nichts zu zeichnen ist. Reine rechnung, damit
        der test sie ohne GL-kontext pruefen kann.
        """
        pulse = 1.0 + (float(self.selection_pulse_amount)
                       * math.sin(self._selection_pulse_phase))
        length = self.ui_px(self.selection_arrow_length_px) * pulse
        half_width = 0.5 * self.ui_px(self.selection_arrow_width_px) * pulse
        gap = self.ui_px(self.selection_gap_px)
        # Das atmen sitzt im ABSTAND, nicht in der pfeilgroesse allein -- bei
        # einem bildfuellenden koerper waeren 7 % von 13 px sonst unsichtbar.
        breathe = length * float(self.selection_pulse_amount) * 2.0 * math.sin(
            self._selection_pulse_phase)

        radius = min(max(float(body_radius_px),
                         self.ui_px(self.selection_min_radius_px)),
                     self.ui_px(self.selection_max_radius_px))
        ring = radius + gap + breathe
        if not math.isfinite(ring) or ring <= 0.0:
            return None

        verts = []
        base = self._selection_spin_phase
        for k in range(4):
            angle = base + k * (math.pi * 0.5)
            dx = math.cos(angle)
            dy = math.sin(angle)
            # Spitze zeigt nach INNEN, auf den koerper.
            tip_x = cx + dx * ring
            tip_y = cy + dy * ring
            back_x = cx + dx * (ring + length)
            back_y = cy + dy * (ring + length)
            # Normale zur pfeilachse, fuer die basisbreite.
            nx = -dy * half_width
            ny = dx * half_width
            verts.append((tip_x, self._ortho_y(tip_y)))
            verts.append((back_x + nx, self._ortho_y(back_y + ny)))
            verts.append((back_x - nx, self._ortho_y(back_y - ny)))
        return verts

    def _draw_selection_marker(self, camera):
        """Vier pfeile um den ausgewaehlten koerper. EIN zeichenaufruf.

        Wird nach dem FXAA-resolve gezeichnet (wie die koerper-beschriftungen):
        ein kantenfilter ueber vier duenne dreiecke verwaescht genau die
        spitzen, die auf den koerper zeigen sollen.
        """
        body = self.selected_body
        if body is None or not self.selection_marker_enabled:
            return
        if self._ortho_vao is None:
            return
        try:
            sx, sy = self._world_to_screen_xy(
                float(body.position.x), float(body.position.y), camera,
                camera_frame_xy=self._frame_camera_xy(camera),
            )
        except Exception:
            return
        if not (math.isfinite(sx) and math.isfinite(sy)):
            return
        # Ausserhalb des bildes gibt es nichts zu markieren. Die marge deckt
        # die pfeile ab, die noch hereinragen koennen.
        reach = self.ui_px(self.selection_arrow_length_px
                           + self.selection_gap_px
                           + self.selection_max_radius_px)
        if not self._is_on_screen(sx, sy, reach):
            return

        verts = self._selection_marker_vertices(
            sx, sy, self._pick_radius_px(body, camera))
        if not verts:
            return
        self._draw_ortho_shape(verts, self.selection_marker_color,
                               moderngl.TRIANGLES)

    def _is_on_screen(self, sx, sy, margin_px):
        return (-margin_px <= sx <= self.width + margin_px and
                -margin_px <= sy <= self.height + margin_px)

    def _visible_window_runs(self, screen_points, margin_px, coords=None):
        return self._build_clipped_polyline_runs(screen_points, margin_px,
                                                 coords=coords)

    def _effective_sampling_tolerance(self, camera):
        scale = abs(float(camera.scale))
        reference_scale = max(self.prediction_sampling_reference_scale, 1e-30)
        zoom_factor = max(1.0, scale / reference_scale)
        tolerance = self.prediction_sampling_tolerance_px / zoom_factor
        tolerance = min(self.prediction_sampling_max_tolerance_px, tolerance)
        return max(self.prediction_sampling_min_tolerance_px, tolerance)

    def _effective_max_segment_step(self, camera):
        scale = abs(float(camera.scale))
        reference_scale = max(self.prediction_sampling_reference_scale, 1e-30)
        zoom_factor = max(1.0, scale / reference_scale)
        step = self.prediction_sampling_max_segment_px / math.sqrt(zoom_factor)
        # Allow smaller max-segment when zoomed in; keep a small floor to
        # avoid degenerate zero-length subdivisions.
        return max(0.5, step)

    def _compact_min_step_indices(self, xs, ys, min_step2):
        """Python-fassung von `_compact_min_step_numba` (fallback ohne numba).

        Gibt die zu behaltenden INDIZES zurueck, nicht die punkte -- so
        arbeitet der zeichenweg auch ohne numba durchgehend auf spalten.
        """
        n = len(xs)
        if n == 0:
            return []
        keep = [0]
        lx = float(xs[0])
        ly = float(ys[0])
        for i in range(1, n):
            sx = float(xs[i])
            sy = float(ys[i])
            dx = sx - lx
            dy = sy - ly
            if dx * dx + dy * dy >= min_step2:
                keep.append(i)
                lx = sx
                ly = sy
        last = keep[-1]
        if xs[last] != xs[n - 1] or ys[last] != ys[n - 1]:
            keep.append(n - 1)
        return keep

    def _max_gap_refine_indices(self, keep_indices, xs, ys, max_seg):
        """Python-fassung von `_max_gap_refine_numba` (fallback ohne numba)."""
        refined = [int(keep_indices[0])]
        for i in range(1, len(keep_indices)):
            start_idx = refined[-1]
            end_idx = int(keep_indices[i])
            if end_idx <= start_idx:
                continue

            seg_dx = float(xs[end_idx]) - float(xs[start_idx])
            seg_dy = float(ys[end_idx]) - float(ys[start_idx])
            seg_len = math.sqrt(seg_dx * seg_dx + seg_dy * seg_dy)

            if seg_len > max_seg:
                steps = max(2, int(math.ceil(seg_len / max_seg)))
                for step_i in range(1, steps):
                    candidate = start_idx + int(round((end_idx - start_idx) * (step_i / steps)))
                    if candidate <= refined[-1]:
                        candidate = refined[-1] + 1
                    if candidate >= end_idx:
                        break
                    refined.append(candidate)

            if end_idx > refined[-1]:
                refined.append(end_idx)

        return np.asarray(refined, dtype=np.int64)

    def _densify_screen_columns(self, xs, ys, max_segment_px):
        """Wie `_densify_screen_run`, aber auf spalten -- gibt (n, 2) zurueck."""
        n = int(len(xs))
        out = np.empty((n, 2), dtype=np.float64)
        if n < 2:
            if n:
                out[0, 0] = xs[0]
                out[0, 1] = ys[0]
            return out

        max_segment = max(0.5, float(max_segment_px))
        if _LINE_KERNELS_OK:
            dx, dy = _densify_numba(
                np.ascontiguousarray(xs, dtype=np.float64),
                np.ascontiguousarray(ys, dtype=np.float64),
                max_segment)
            dense = np.empty((dx.shape[0], 2), dtype=np.float64)
            dense[:, 0] = dx
            dense[:, 1] = dy
            return dense

        run = np.empty((n, 2), dtype=np.float64)
        run[:, 0] = xs
        run[:, 1] = ys
        return np.asarray(
            self._densify_screen_run([tuple(p) for p in run], max_segment_px),
            dtype=np.float64)

    def _densify_screen_run(self, run, max_segment_px):
        if len(run) < 2:
            return run

        max_segment = max(0.5, float(max_segment_px))
        dense = [run[0]]
        for i in range(len(run) - 1):
            x0, y0 = run[i]
            x1, y1 = run[i + 1]
            dx = x1 - x0
            dy = y1 - y0
            seg_len = math.sqrt(dx * dx + dy * dy)

            if seg_len > max_segment:
                parts = int(math.ceil(seg_len / max_segment))
                parts = max(2, min(256, parts))
                for p in range(1, parts):
                    t = p / parts
                    dense.append((x0 + dx * t, y0 + dy * t))

            dense.append((x1, y1))

        return dense

    def _prediction_point_key(self, points, index):
        try:
            point = points[index]
            if hasattr(point, 'x') and hasattr(point, 'y'):
                return (float(point.x), float(point.y), None)
            t = None
            try:
                if hasattr(point, '__len__') and len(point) >= 3:
                    t = float(point[2])
            except Exception:
                t = None
            return (float(point[0]), float(point[1]), t)
        except Exception:
            return None

    def _make_prediction_line_cache_key(self, path_points, input_count, camera, anchor_world):
        shape = getattr(path_points, 'shape', None)
        if shape is not None:
            try:
                shape_key = tuple(int(v) for v in shape)
            except Exception:
                shape_key = (int(input_count),)
        else:
            shape_key = (int(input_count),)

        anchor_key = None
        if anchor_world is not None:
            try:
                anchor_key = (float(anchor_world[0]), float(anchor_world[1]))
            except Exception:
                anchor_key = None

        active_frame = self._active_frame()
        return (
            id(path_points),
            shape_key,
            int(input_count),
            self._prediction_point_key(path_points, 0),
            self._prediction_point_key(path_points, input_count - 1),
            float(camera.position.x),
            float(camera.position.y),
            float(camera.scale),
            int(self.width),
            int(self.height),
            id(active_frame),
            getattr(active_frame, 'label', None),
            id(self._target_frame),
            self._target_frame_label,
            self._plotting_frame_label,
            float(self._frame_time_s),
            anchor_key,
            float(self.prediction_sampling_tolerance_px),
            float(self.prediction_sampling_min_step_px),
            float(self.prediction_sampling_max_points),
            float(self.prediction_sampling_max_segment_px),
            float(self.prediction_sampling_reference_scale),
            float(self.prediction_visibility_margin_px),
            int(self.prediction_render_max_raw_scan),
            int(self.prediction_render_max_draw_points),
            None if self.prediction_render_max_world_length is None else float(self.prediction_render_max_world_length),
            None if self.prediction_render_max_screen_length_px is None else float(self.prediction_render_max_screen_length_px),
            bool(self.prediction_hermite_enabled),
            float(self.prediction_detail_scale),
            int(self.prediction_hermite_max_subdiv),
            tuple(float(v) for v in self.prediction_error_ladder_m),
        )

    def _prediction_scan_indices(self, raw_count, stats):
        try:
            max_scan = int(self.prediction_render_max_raw_scan)
        except Exception:
            max_scan = 0
        indices = self._iter_prediction_indices_evenly(raw_count, max_scan)
        if len(indices) >= 2:
            stride_est = max(1, int(round((int(raw_count) - 1) / float(len(indices) - 1))))
        else:
            stride_est = 1
        stats['raw_stride'] = stride_est
        stats['skipped_by_stride'] = max(0, int(raw_count) - len(indices))
        return indices

    def _iter_prediction_indices_evenly(self, count, max_scan):
        """Gleichmaessige stichprobe der rohpunkte -- GEMERKT, nicht neu gebaut.

        Das ergebnis haengt einzig an (count, max_scan). Der cache greift
        aber NUR, solange beide gleich bleiben -- und `count` aendert sich im
        zeitraffer jeden frame, weil der halt die punkteliste vorn beschneidet
        (siehe Predictor._hold_advance). Deshalb muss auch der neubau billig
        sein; er laeuft ueber numpy statt ueber eine Python-schleife.
        """
        count = int(count)
        max_scan = int(max_scan)

        key = (count, max_scan)
        cached = getattr(self, '_prediction_indices_cache', None)
        if cached is not None and cached[0] == key:
            return cached[1]

        indices = self._build_prediction_indices(count, max_scan)
        self._prediction_indices_cache = (key, indices)
        return indices

    def _build_prediction_indices(self, count, max_scan):
        if count <= 0:
            return []

        if max_scan <= 0 or count <= max_scan:
            return list(range(count))

        if max_scan == 1:
            return [0]

        step = (count - 1) / float(max_scan - 1)

        if np is not None:
            # np.rint rundet wie Pythons round() zur GERADEN zahl hin, die
            # stichprobe ist damit dieselbe wie in der schleife unten.
            idx = np.rint(np.arange(max_scan, dtype=np.float64) * step)
            idx = idx.astype(np.int64)
            np.clip(idx, 0, count - 1, out=idx)
            # Nur AUFEINANDERFOLGENDE wiederholungen fallen weg -- genau das
            # tut die schleife mit ihrem `last`.
            keep = np.empty(idx.shape, dtype=bool)
            keep[0] = True
            np.not_equal(idx[1:], idx[:-1], out=keep[1:])
            return idx[keep]

        indices = []
        last = -1
        for i in range(max_scan):
            idx = int(round(i * step))
            idx = max(0, min(count - 1, idx))
            if idx != last:
                indices.append(idx)
                last = idx
        return indices

    def _cap_runs_by_screen_length(self, runs, max_screen_length_px, stats):
        if max_screen_length_px is None:
            return runs
        try:
            max_length = float(max_screen_length_px)
        except Exception:
            return runs
        if max_length <= 0.0:
            stats['clipped_or_rejected'] = stats.get('clipped_or_rejected', 0) + sum(len(run) for run in runs)
            return []

        run_lengths = []
        total_length = 0.0
        for run in runs:
            if len(run) < 2:
                run_lengths.append(0.0)
                continue
            length = 0.0
            for i in range(len(run) - 1):
                lx, ly = run[i]
                sx, sy = run[i + 1]
                dx = float(sx) - float(lx)
                dy = float(sy) - float(ly)
                length += math.sqrt(dx * dx + dy * dy)
            run_lengths.append(length)
            total_length += length

        if total_length <= max_length:
            return runs

        capped = []
        rejected = 0
        for run, run_length in zip(runs, run_lengths):
            if len(run) < 2 or run_length <= 1e-12:
                rejected += len(run)
                continue

            remaining = max_length * (run_length / total_length)
            if remaining <= 1e-12:
                rejected += len(run)
                continue

            current = [run[0]]
            for i in range(len(run) - 1):
                lx, ly = current[-1]
                sx, sy = run[i + 1]
                dx = float(sx) - float(lx)
                dy = float(sy) - float(ly)
                seg_len = math.sqrt(dx * dx + dy * dy)
                if seg_len <= remaining:
                    current.append((sx, sy))
                    remaining -= seg_len
                    continue
                if seg_len > 1e-12 and remaining > 0.0:
                    frac = remaining / seg_len
                    current.append((lx + dx * frac, ly + dy * frac))
                break
            if len(current) >= 2:
                capped.append(current)
            rejected += max(0, len(run) - len(current))

        stats['clipped_or_rejected'] = stats.get('clipped_or_rejected', 0) + rejected
        return capped

    def _cap_runs_by_point_budget(self, runs, max_points, stats):
        capped = self._limit_polyline_runs_evenly(runs, max_points)
        rejected = max(0, sum(len(run) for run in runs) - sum(len(run) for run in capped))
        stats['clipped_or_rejected'] = stats.get('clipped_or_rejected', 0) + rejected
        return capped

    def _limit_polyline_runs_evenly(self, runs, max_points):
        max_points = int(max_points)
        if max_points <= 0:
            return []

        total = sum(len(run) for run in runs)
        if total <= max_points:
            return runs

        limited = []
        points_left = max_points
        runs_left = len(runs)

        for run in runs:
            if runs_left <= 0 or points_left <= 1:
                break

            budget = max(2, points_left // runs_left)
            if len(run) <= budget:
                limited.append(run)
                points_left -= len(run)
            elif np is not None and isinstance(run, np.ndarray):
                # Gleiche stichprobe wie unten, nur als index-rechnung.
                # np.round rundet -- wie Pythons round() -- die haelfte zur
                # geraden zahl, die gewaehlten indizes sind also dieselben.
                step = (len(run) - 1) / float(budget - 1)
                idx = np.round(np.arange(budget, dtype=np.float64) * step)
                idx = np.clip(idx, 0.0, float(len(run) - 1)).astype(np.int64)
                if idx.shape[0] > 1:
                    unique = np.empty(idx.shape[0], dtype=bool)
                    unique[0] = True
                    np.not_equal(idx[1:], idx[:-1], out=unique[1:])
                    idx = idx[unique]
                if idx.shape[0] >= 2:
                    limited.append(run[idx])
                points_left -= int(idx.shape[0])
            else:
                step = (len(run) - 1) / float(budget - 1)
                sampled = []
                last = -1
                for i in range(budget):
                    idx = int(round(i * step))
                    idx = max(0, min(len(run) - 1, idx))
                    if idx != last:
                        sampled.append(run[idx])
                        last = idx
                if len(sampled) >= 2:
                    limited.append(sampled)
                points_left -= len(sampled)

            runs_left -= 1

        return limited

    def draw_prediction(self, path_points, camera, anchor_world=None, predictor=None):

        input_count = self._points_count(path_points)
        stats = {
            'raw_in': int(input_count),
            'raw_points': int(input_count),
            'scanned': 0,
            'scanned_points': 0,
            'visible': 0,
            'runs': 0,
            'draw_points': 0,
            'drawn': 0,
            'skipped_by_stride': 0,
            'clipped_or_rejected': 0,
            'prepare_ms': 0.0,
            'draw_ms': 0.0,
            'cache_hit': False,
        }
        if input_count == 0:
            self.debug_info['prediction_points_in'] = 0
            self.debug_info['prediction_points_drawn'] = 0
            self._last_prediction_render_stats = stats
            return

        # Blending ist global aktiv (ctx.enable in _init_opengl); die alten
        # textur-/blend-state-resets der fixed-function-pipeline entfallen.
        prepare_t0 = time.perf_counter()
        half_w = self.width * 0.5
        half_h = self.height * 0.5
        camera_frame_xy = self._frame_camera_xy(camera)
        self._debug_prediction_frame_transform(path_points, predictor=predictor)

        # debug-ausgabe: schiff-welt-position und ersten predictor-punkt anzeigen
        try:
            pred0_x, pred0_y = self._point_xy(path_points[0])
            ship_world_x, ship_world_y = (float(anchor_world[0]), float(anchor_world[1])) if anchor_world is not None else (pred0_x, pred0_y)
            if self.debug_predictor:
                print(f"PRED_DBG_POS: ship=({ship_world_x:.6e},{ship_world_y:.6e}) predictor_first=({pred0_x:.6e},{pred0_y:.6e})")
        except Exception:
            pass

        effective_tolerance = self._effective_sampling_tolerance(camera)
        effective_min_step = max(0.05, min(self.prediction_sampling_min_step_px, effective_tolerance * 0.6))
        effective_max_segment = self._effective_max_segment_step(camera)
        max_draw_points = max(2, min(int(self.prediction_sampling_max_points), int(self.prediction_render_max_draw_points)))

        # EIN BUDGET FUER DIE GANZE KETTE. Der zeichenweg hat drei stellen, an
        # denen er punkte wegwirft (min-schritt-verdichtung, RDP, run-kappung)
        # und eine, an der er welche setzt (kubische unterteilung). Solange die
        # wegwerfenden ihre eigene, aelteren zoom-heuristik folgende toleranz
        # benutzen, macht die eine haelfte zunichte, was die andere aufbaut:
        # gemessen 1990 m abweichung bei einer zusage von 1000 m, weil die
        # RDP-toleranz bis 0.25 px gehen darf -- bei 4.4e-5 px/m sind das
        # 5700 m. Also bekommen alle stufen ihren anteil an DERSELBEN zusage.
        self._prediction_detail_budget = None
        if self.prediction_hermite_enabled:
            budget = self._prediction_error_budget(camera)
            if budget is not None:
                self._prediction_detail_budget = budget
                eps_px = budget[1]
                # Die anteile addieren sich im schlimmsten fall, also muessen
                # sie zusammen unter 1 bleiben: 0.5 unterteilung + 0.25 RDP +
                # 0.1 verdichtung = 0.85. Mit 0.5/0.5/0.25 (summe 1.25) lag die
                # gemessene abweichung bei 1.12 der zusage -- knapp darueber.
                effective_tolerance = max(1e-3, min(effective_tolerance, eps_px * 0.25))
                effective_min_step = max(1e-3, min(effective_min_step, eps_px * 0.1))

        cache_key = self._make_prediction_line_cache_key(path_points, input_count, camera, anchor_world)
        if cache_key == self._prediction_line_cache_key_value and self._prediction_line_cache_points is not None:
            sampled_runs = self._prediction_line_cache_points
            stats.update(dict(self._prediction_line_cache_stats))
            stats['raw_in'] = int(input_count)
            stats['raw_points'] = int(input_count)
            stats['prepare_ms'] = (time.perf_counter() - prepare_t0) * 1000.0
            stats['cache_hit'] = True
        else:
            # Bewegte origin-frames (z.B. Erde): origin-position über das
            # predictor-zeitfenster interpolieren statt pro punkt propagieren.
            # Auf das aktive frame begrenzt und danach wieder gelöscht, damit
            # körper/spuren (aktuelle zeit) exakt transformiert bleiben.
            active_frame = self._active_frame()
            interp_window_set = False
            try:
                try:
                    p_first = path_points[0]
                    p_last = path_points[input_count - 1]
                    t_first = float(p_first[2]) if hasattr(p_first, '__len__') and len(p_first) >= 3 else None
                    t_last = float(p_last[2]) if hasattr(p_last, '__len__') and len(p_last) >= 3 else None
                except Exception:
                    t_first = t_last = None
                if t_first is not None and t_last is not None and t_last > t_first:
                    active_frame.set_origin_interp_window(t_first, t_last, int(input_count))
                    interp_window_set = True
                sampled_runs = self._adaptive_prediction_screen_points(
                    path_points,
                    camera,
                    tolerance_px=effective_tolerance,
                    min_step_px=effective_min_step,
                    max_segment_px=effective_max_segment,
                    max_points=max_draw_points,
                    margin_px=self.prediction_visibility_margin_px,
                    anchor_world=anchor_world,
                    stats=stats,
                    camera_frame_xy=camera_frame_xy,
                )
            finally:
                if interp_window_set:
                    active_frame.set_origin_interp_window(0.0, 0.0, 0)
            stats['prepare_ms'] = (time.perf_counter() - prepare_t0) * 1000.0

        self.debug_info['prediction_points_in'] = input_count
        self.debug_info['prediction_points_drawn'] = sum(len(run) for run in sampled_runs)

        # Store small sample for debugging and optionally print it.
        try:
            if sampled_runs and len(sampled_runs[0]) > 0:
                sample_n = min(5, len(sampled_runs[0]))
                screen_samples = [sampled_runs[0][i] for i in range(sample_n)]
                frame_samples = []
                for sx, sy in screen_samples:
                    fx = camera_frame_xy[0] + (sx - half_w) / float(camera.scale)
                    fy = camera_frame_xy[1] - (sy - half_h) / float(camera.scale)
                    frame_samples.append((fx, fy))
                self.debug_info['prediction_sample_screen'] = screen_samples
                self.debug_info['prediction_sample_frame'] = frame_samples
                if self.debug_predictor:
                    print('PRED_DBG: in=', input_count, 'drawn=', self.debug_info['prediction_points_drawn'])
                    print('PRED_DBG: screen_samples=', screen_samples)
                    print('PRED_DBG: frame_samples=', frame_samples)
        except Exception:
            pass

        if len(sampled_runs) == 0:
            stats['drawn'] = 0
            self._last_prediction_render_stats = stats
            return

        sampled_runs = self._cap_runs_by_point_budget(sampled_runs, max_draw_points, stats)
        self._prediction_line_cache_key_value = cache_key
        self._prediction_line_cache_points = sampled_runs
        self._prediction_line_cache_stats = dict(stats)
        draw_t0 = time.perf_counter()
        for run in sampled_runs:
            if len(run) < 2:
                continue
            self._draw_polyline(run, color=(1.0, 1.0, 1.0, 0.6), width=2.0)
        self._draw_apsis_markers(predictor, camera, camera_frame_xy)
        stats['draw_ms'] = (time.perf_counter() - draw_t0) * 1000.0
        stats['drawn'] = sum(len(run) for run in sampled_runs)
        stats['draw_points'] = int(stats['drawn'])
        stats['runs'] = len(sampled_runs)
        self.debug_info['prediction_points_drawn'] = int(stats['drawn'])

        # WAS DIE LINIE WIRKLICH KANN, NICHT WAS SIE VERSPRICHT. Die sprosse
        # der leiter ist das ziel; der interpolations-boden des predictors
        # (punktabstand^4 / 384 R^3) kann darueber liegen, und dann ist ER die
        # erreichte genauigkeit. Beides anzuzeigen ist der einzige weg, eine
        # zugesagte praezision nicht still zu erfinden.
        floor = None
        try:
            get_floor = getattr(predictor, 'interpolation_error_floor', None)
            if get_floor is not None:
                floor = get_floor()
        except Exception:
            floor = None
        stats['detail_floor_m'] = floor
        eps_m = stats.get('detail_eps_m')
        if eps_m is not None:
            achieved = eps_m if floor is None else max(eps_m, floor)
            stats['detail_achieved_m'] = achieved
            self.debug_info['prediction_detail_target_m'] = eps_m
            self.debug_info['prediction_detail_achieved_m'] = achieved
            self.debug_info['prediction_detail_added'] = int(
                stats.get('hermite_added', 0))

        self._last_prediction_render_stats = stats

    def _format_apsis_distance(self, r):
        if r >= 1e9:
            return f"{r / 1e9:.2f}Gm"
        if r >= 1e6:
            return f"{r / 1e6:.2f}Mm"
        if r >= 1e3:
            return f"{r / 1e3:.1f}km"
        return f"{r:.0f}m"

    def _draw_apsis_markers(self, predictor, camera, camera_frame_xy=None):
        """Zeichnet apoapsis/periapsis-marker des predictors auf die linie.

        Marker kommen als (m, 5)-array (x, y, t_abs, kind, r) aus
        predictor.get_apsis_markers(); die transformation nutzt die
        zeitabhängige frame-transformation, damit die marker in bewegten
        plot-frames auf der gezeichneten linie bleiben.
        """
        if predictor is None or not self.show_apsis_markers:
            return
        get_markers = getattr(predictor, 'get_apsis_markers', None)
        if get_markers is None:
            return
        try:
            markers = get_markers()
        except Exception:
            return
        count = self._points_count(markers)
        if count == 0:
            return

        r_px = float(self.apsis_marker_radius_px)

        # AUSBLENDEN NACH SCHIRMGROESSE DER BAHN, NICHT NACH ZOOM-SCHWELLE.
        # `m[4]` ist der apsis-abstand zum bezugskoerper in metern, `* scale`
        # also der apsis-radius in PIXELN -- ein direktes mass dafuer, wie
        # gross die bahn am schirm ist. Wird sie klein, ruecken Pe/Ap an die
        # schiffs- und die Erde-marke heran; das smoothstep zwischen
        # `fade_min_px` und `fade_full_px` blendet sie dann sauber weg, statt
        # sie uebereinanderzustapeln. Damit ist die alte "ein draw je farbe"-
        # buendelung hin (jeder marker hat jetzt seine eigene deckkraft) --
        # bei real 1 Pe + 1 Ap, selten je zwei, ist das ein draw je marker
        # und faellt nicht ins gewicht.
        scale = abs(float(getattr(camera, 'scale', 0.0)))
        fade_min = float(self.apsis_marker_fade_min_px)
        fade_full = float(self.apsis_marker_fade_full_px)

        for i in range(count):
            try:
                m = markers[i]
                wx = float(m[0])
                wy = float(m[1])
                t_abs = float(m[2])
                is_apo = float(m[3]) >= 0.5
                dist = float(m[4])
            except Exception:
                continue

            size_px = dist * scale
            if fade_full > fade_min:
                u = (size_px - fade_min) / (fade_full - fade_min)
                u = max(0.0, min(1.0, u))
                alpha_mult = u * u * (3.0 - 2.0 * u)
            else:
                alpha_mult = 1.0 if size_px >= fade_min else 0.0
            if alpha_mult <= 0.003:
                continue

            sx, sy = self._world_to_screen_xy_at_time(wx, wy, camera, t_abs, camera_frame_xy)
            if not (math.isfinite(sx) and math.isfinite(sy)):
                continue
            if not self._is_on_screen(sx, sy, 32.0):
                continue

            label = "Ap" if is_apo else "Pe"
            base = (0.45, 0.75, 1.0) if is_apo else (1.0, 0.62, 0.25)
            col = (base[0], base[1], base[2], 0.95 * alpha_mult)

            # Nord -> Ost -> Sued -> West -> Nord, als vier strecken.
            north = (sx, sy - r_px)
            east = (sx + r_px, sy)
            south = (sx, sy + r_px)
            west = (sx - r_px, sy)
            self._draw_line_segments(
                (north, east, east, south, south, west, west, north),
                color=col, width=2.0)

            text = f"{label} {self._format_apsis_distance(dist)}"
            try:
                # Diamant + linie laufen über den line-shader (top-down, sy
                # wächst nach unten), text über die ortho-konvention
                # (y nach oben). _blit_text_topdown rechnet das um.
                entry = self._get_label_texture(text, self.font_small)
                tw = float(entry[1]) if entry else 0.0
                label_x = sx - tw / 2.0
                self._blit_text_topdown(text, label_x, sy + r_px + 4.0,
                                        self.font_small,
                                        color=(1.0, 1.0, 1.0, alpha_mult))
            except Exception:
                pass

    def _squared_point_line_distance(self, px, py, ax, ay, bx, by):
        abx = bx - ax
        aby = by - ay
        ab2 = abx * abx + aby * aby
        if ab2 <= 1e-18:
            dx = px - ax
            dy = py - ay
            return dx * dx + dy * dy

        apx = px - ax
        apy = py - ay
        t = (apx * abx + apy * aby) / ab2
        t = max(0.0, min(1.0, t))
        proj_x = ax + t * abx
        proj_y = ay + t * aby
        dx = px - proj_x
        dy = py - proj_y
        return dx * dx + dy * dy

    def _rdp_indices(self, points, tolerance_px):
        n = len(points)
        if n <= 2:
            return [0, n - 1] if n == 2 else [0]

        tol2 = tolerance_px * tolerance_px
        keep = [False] * n
        keep[0] = True
        keep[-1] = True
        stack = [(0, n - 1)]

        while stack:
            start, end = stack.pop()
            if end <= start + 1:
                continue

            ax, ay = points[start]
            bx, by = points[end]
            max_d2 = -1.0
            index = -1

            for i in range(start + 1, end):
                px, py = points[i]
                d2 = self._squared_point_line_distance(px, py, ax, ay, bx, by)
                if d2 > max_d2:
                    max_d2 = d2
                    index = i

            if max_d2 > tol2 and index != -1:
                keep[index] = True
                stack.append((start, index))
                stack.append((index, end))

        return [i for i, k in enumerate(keep) if k]

    def _adaptive_prediction_screen_points(self,
                                           path_points,
                                           camera,
                                           tolerance_px,
                                           min_step_px,
                                           max_segment_px,
                                           max_points,
                                           margin_px,
                                           anchor_world=None,
                                           stats=None,
                                           camera_frame_xy=None):
        if stats is None:
            stats = {}
        half_w = self.width * 0.5
        half_h = self.height * 0.5
        if camera_frame_xy is None:
            camera_frame_xy = self._frame_camera_xy(camera)
        scale = float(camera.scale)

        screen_points = []

        raw_count = len(path_points)
        indices = self._prediction_scan_indices(raw_count, stats)
        max_world_length = self.prediction_render_max_world_length
        try:
            max_world_length = None if max_world_length is None else max(0.0, float(max_world_length))
        except Exception:
            max_world_length = None

        # Schneller weg: alle punkte auf einmal projizieren. Greift nur, wenn
        # keine weltlaengen-begrenzung aktiv ist (die braucht den laufenden
        # summenwert und damit die schleife) und der rahmen eine stapel-
        # transformation anbietet. Sonst faellt es auf die schleife darunter
        # zurueck -- dieselbe rechnung, nur langsam.
        batch = None
        if max_world_length is None:
            batch = self._project_prediction_batch(
                path_points, indices, camera, camera_frame_xy, margin_px)

        if batch is not None:
            # `screen_points` ist auf diesem weg IMMER None: die spalten in
            # `coords` sind die punkte. Die tupel-liste, die hier frueher
            # entstand, wurde weiter unten ohnehin wieder in arrays
            # zurueckverwandelt (gemessen 4000 tupel je frame, nur um sie
            # danach wegzuwerfen).
            screen_points, visible_count, coords = batch

            # ZWISCHEN den groben stuetzstellen kubisch nachlegen -- so fein,
            # wie der bildschirm es zeigt, und nur dort, wo etwas zu sehen
            # ist. Schlaegt das fehl (kein tangenten-paar, kein stapel-
            # rahmen, kein budget), bleibt es bei den groben punkten und die
            # linie sieht aus wie vorher.
            budget = getattr(self, '_prediction_detail_budget', None)
            if budget is not None:
                eps_m, eps_px, rung = budget
                stats['detail_eps_m'] = eps_m
                stats['detail_eps_px'] = eps_px
                stats['detail_rung'] = rung
                refined = self._hermite_refine_world(
                    path_points, indices, coords, camera,
                    eps_px * 0.5, margin_px,
                    int(self.prediction_render_max_draw_points), stats)
                if refined is not None:
                    dense = self._project_prediction_batch(
                        refined, np.arange(refined.shape[0], dtype=np.int64),
                        camera, camera_frame_xy, margin_px)
                    if dense is not None:
                        screen_points, visible_count, coords = dense

            scanned_count = (len(coords[0]) if screen_points is None
                             else len(screen_points))
            stats['scanned'] = stats.get('scanned', 0) + scanned_count
            stats['scanned_points'] = stats.get('scanned_points', 0) + scanned_count
            stats['visible'] = stats.get('visible', 0) + visible_count
            return self._runs_from_screen_points(
                screen_points, camera, tolerance_px, min_step_px,
                max_segment_px, max_points, margin_px, stats, coords=coords)

        prev_world = None
        prev_time = None
        world_accum = 0.0

        for i in indices:
            point = path_points[i]
            px, py = self._point_xy(point)

            # If point includes timestamp (x,y,t), use time-aware projection.
            sample_time = None
            try:
                if hasattr(point, '__len__') and len(point) >= 3:
                    sample_time = float(point[2])
            except Exception:
                sample_time = None

            stop_after_point = False
            if max_world_length is not None and prev_world is not None:
                seg_dx_world = px - prev_world[0]
                seg_dy_world = py - prev_world[1]
                seg_len_world = math.sqrt(seg_dx_world * seg_dx_world + seg_dy_world * seg_dy_world)
                if world_accum + seg_len_world > max_world_length:
                    remaining_world = max_world_length - world_accum
                    if seg_len_world > 1e-12 and remaining_world > 0.0:
                        frac = remaining_world / seg_len_world
                        px = prev_world[0] + seg_dx_world * frac
                        py = prev_world[1] + seg_dy_world * frac
                        if sample_time is not None and prev_time is not None:
                            sample_time = prev_time + (sample_time - prev_time) * frac
                    else:
                        px, py = prev_world
                        sample_time = prev_time
                    stop_after_point = True
                else:
                    world_accum += seg_len_world

            if sample_time is not None:
                sx, sy = self._world_to_screen_xy_at_time(
                    px,
                    py,
                    camera,
                    sample_time,
                    camera_frame_xy=camera_frame_xy,
                )
            else:
                frame_x, frame_y = self._frame_transform_xy(px, py)
                sx = half_w + (frame_x - camera_frame_xy[0]) * scale
                sy = half_h - (frame_y - camera_frame_xy[1]) * scale
            stats['scanned'] = stats.get('scanned', 0) + 1
            stats['scanned_points'] = stats.get('scanned_points', 0) + 1

            near_visible = self._is_on_screen(sx, sy, margin_px)
            if near_visible:
                stats['visible'] = stats.get('visible', 0) + 1

            screen_points.append((sx, sy))

            prev_world = (px, py)
            prev_time = sample_time
            if stop_after_point:
                stats['clipped_or_rejected'] = stats.get('clipped_or_rejected', 0) + max(0, raw_count - i - 1)
                break

        return self._runs_from_screen_points(
            screen_points, camera, tolerance_px, min_step_px, max_segment_px,
            max_points, margin_px, stats)

    def _project_prediction_batch(self, path_points, indices, camera,
                                  camera_frame_xy, margin_px):
        """Alle stichproben-punkte in EINEM rutsch projizieren.

        Gibt ``(None, sichtbar_anzahl, (sx, sy))`` zurueck oder None, wenn
        der schnelle weg nicht anwendbar ist. Rechnerisch identisch zur
        schleife in _adaptive_prediction_screen_points -- dieselbe rahmen-
        transformation, derselbe massstab, dieselbe y-spiegelung.

        Der erste eintrag ist bewusst None: die punkte leben ab hier nur
        noch in den spalten (sx, sy). Die frueher hier gebaute liste aus
        (x, y)-tupeln kostete bei 4000 punkten je frame mehr als die
        projektion selbst und wurde vom zeichenweg sofort wieder in arrays
        zurueckverwandelt.

        Motivation: gemessen lag die punktweise projektion bei 5.6 ms je
        frame, praktisch alles davon Python-aufruf-overhead ueber 3000
        punkte, von denen am ende 196 gezeichnet werden.
        """
        if np is None:
            return None
        if not isinstance(path_points, np.ndarray):
            return None
        if path_points.ndim != 2 or path_points.shape[1] < 3:
            return None
        if len(indices) == 0:
            empty = np.empty(0, dtype=np.float64)
            return (None, 0, (empty, empty))

        frame = self._active_frame()
        transform = getattr(frame, 'to_this_frame_xy_arrays', None)
        if transform is None:
            return None

        idx = np.asarray(indices, dtype=np.int64)
        sub = path_points[idx]
        xs = np.ascontiguousarray(sub[:, 0], dtype=np.float64)
        ys = np.ascontiguousarray(sub[:, 1], dtype=np.float64)
        ts = np.ascontiguousarray(sub[:, 2], dtype=np.float64)

        try:
            transformed = transform(ts, xs, ys)
        except Exception:
            return None
        if transformed is None:
            return None
        frame_x, frame_y = transformed

        scale = float(camera.scale)
        sx = self.width * 0.5 + (frame_x - camera_frame_xy[0]) * scale
        sy = self.height * 0.5 - (frame_y - camera_frame_xy[1]) * scale

        # Gleiche schranke wie _is_on_screen -- rein diagnostisch (stats).
        margin = float(margin_px)
        visible = int(np.count_nonzero(
            (sx >= -margin) & (sx <= self.width + margin)
            & (sy >= -margin) & (sy <= self.height + margin)))
        return (None, visible, (sx, sy))

    # ------------------------------------------------------------------
    # Aufloesungsgetriebene verfeinerung
    # ------------------------------------------------------------------

    def _prediction_error_budget(self, camera):
        """Erlaubte abweichung der gezeichneten linie -- in metern und pixeln.

        Der bildschirm gibt den wunsch vor (`eps_px / view_scale`), die
        toleranz-leiter quantisiert ihn. Gewaehlt wird die GROESSTE sprosse,
        die den wunsch noch einhaelt; darunter/darueber wird geklemmt. Damit
        ist die zugesagte genauigkeit nie schlechter als angefordert, und sie
        aendert sich nur an wenigen diskreten zoom-schwellen statt stetig.

        Rueckgabe: ``(eps_m, eps_px, rung_index)`` oder ``None``.
        """
        try:
            scale = abs(float(camera.scale))
        except Exception:
            return None
        if not (scale > 0.0) or not math.isfinite(scale):
            return None

        try:
            detail = float(self.prediction_detail_scale)
        except Exception:
            detail = 1.0
        detail = max(1e-3, detail)
        eps_px = 0.3 / detail

        ladder = self.prediction_error_ladder_m
        try:
            rungs = sorted(float(v) for v in ladder if float(v) > 0.0)
        except Exception:
            rungs = []
        if not rungs:
            return (eps_px / scale, eps_px, -1)

        wanted_m = eps_px / scale
        index = 0
        for i, rung in enumerate(rungs):
            if rung <= wanted_m:
                index = i
            else:
                break
        if wanted_m < rungs[0]:
            index = 0
        eps_m = rungs[index]
        # Die sprosse ist das versprechen; die pixel-toleranz muss sich ihr
        # beugen, sonst wird feiner gezeichnet als zugesagt (kostet) oder
        # groeber (bricht das versprechen).
        return (eps_m, eps_m * scale, index)

    def _hermite_refine_world(self, path_points, indices, coords, camera,
                              eps_px, margin_px, budget, stats):
        """Kubische zwischenpunkte setzen -- nur sichtbar, nur so fein wie noetig.

        Arbeitet in WELTKOORDINATEN und gibt ein dichteres ``(m, 3)``-array
        ``[x, y, t]`` zurueck, das anschliessend durch dieselbe
        stapel-projektion laeuft wie die groben punkte. In *screen*-space zu
        unterteilen waere falsch: jeder zwischenpunkt gehoert zu einer eigenen
        zeit, und ein bewegter oder rotierender plot-rahmen bildet ihn deshalb
        anders ab als seine nachbarn.

        Die unterteilungszahl je segment kommt aus der flachheits-schranke fuer
        kubische kurven (siehe unten). Das ist eine heuristik fuer die ANZAHL --
        die gezeichneten punkte selbst werden einzeln zu ihrer eigenen zeit
        projiziert und sind damit exakt.

        Rueckgabe: ``(m, 3)``-array oder ``None`` (dann bleibt alles wie bisher).
        """
        if np is None or not self.prediction_hermite_enabled:
            return None
        if not isinstance(path_points, np.ndarray) or path_points.ndim != 2:
            return None
        # Ohne die geschwindigkeits-spalten gibt es keine tangente und damit
        # nichts zu interpolieren.
        if path_points.shape[1] < 5:
            return None
        if coords is None:
            return None

        idx = np.asarray(indices, dtype=np.int64)
        if idx.size < 2:
            return None

        sx, sy = coords
        if sx.shape[0] != idx.shape[0]:
            return None

        sub = path_points[idx]
        x0 = sub[:-1, 0]; y0 = sub[:-1, 1]; t0 = sub[:-1, 2]
        x1 = sub[1:, 0];  y1 = sub[1:, 1];  t1 = sub[1:, 2]
        vx0 = sub[:-1, 3]; vy0 = sub[:-1, 4]
        vx1 = sub[1:, 3];  vy1 = sub[1:, 4]
        dt = t1 - t0

        sx0 = sx[:-1]; sy0 = sy[:-1]
        sx1 = sx[1:];  sy1 = sy[1:]

        third = dt / 3.0

        # Ohne endliche tangente an BEIDEN enden gibt es kein polynom -- der
        # abschnitt bleibt dann eine gerade. Genau dafuer schreiben die
        # sehnen-kernel NaN in die geschwindigkeitsspalten.
        usable = (np.isfinite(vx0) & np.isfinite(vy0)
                  & np.isfinite(vx1) & np.isfinite(vy1)
                  & np.isfinite(dt) & (dt > 0.0))
        if not np.any(usable):
            return None

        # SICHTBARKEIT ZUERST, DANN RECHNEN. Die flachheits-schaetzung kostet
        # zwei zusaetzliche rahmen-transformationen je segment -- bei 3000
        # segmenten gemessen 2.3 ms je frame, waehrend am ende fuenf punkte
        # dazukamen, weil nur ~200 segmente ueberhaupt im bild lagen. Die
        # vorauswahl laeuft deshalb allein auf den schon vorhandenen
        # bildschirm-endpunkten.
        #
        # Als grosszuegige schranke fuer die auslenkung dient die sehnenlaenge
        # selbst: eine kubische kurve liegt in der konvexen huelle ihrer
        # kontrollpunkte, und die liegen bei einer glatten bahn rund eine
        # drittel sehne neben den endpunkten.
        margin = float(margin_px)
        chord = np.hypot(sx1 - sx0, sy1 - sy0)
        chord = np.where(np.isfinite(chord), chord, 0.0)
        lo_x = np.minimum(sx0, sx1) - chord
        hi_x = np.maximum(sx0, sx1) + chord
        lo_y = np.minimum(sy0, sy1) - chord
        hi_y = np.maximum(sy0, sy1) + chord
        visible = ((hi_x >= -margin) & (lo_x <= self.width + margin)
                   & (hi_y >= -margin) & (lo_y <= self.height + margin))
        candidate = visible & usable
        if not np.any(candidate):
            return None

        # Zweite differenzen der Bezier-kontrollpunkte
        # (b0 = p0, b1 = p0 + v0*dt/3, b2 = p1 - v1*dt/3, b3 = p1), ausmultipliziert
        # -- in WELTKOORDINATEN und dann mit dem massstab in pixel umgerechnet,
        # statt die kontrollpunkte eigens auf den schirm zu projizieren.
        #
        # Das darf man, weil alle plot-rahmen STARR sind (verschiebung plus
        # drehung): beides laesst laengen unveraendert, und eine zweite
        # differenz ist eine laenge. Gemessen kostete die eigene projektion
        # zwei zusaetzliche rahmen-transformationen ueber ~3000 segmente und
        # damit 1.9 ms je frame -- fuer eine ZAHL, die ohnehin nur die
        # unterteilungsstufe waehlt.
        #
        # Nicht erfasst wird die zusaetzliche kruemmung, die ein ROTIERENDER
        # rahmen ueber die dauer eines segments selbst erzeugt. Fuer den
        # Erde-Sonne-richtungsrahmen sind das ~0.14 m gegen 0.83 m echte
        # woelbung; erst bei sehr schnell drehenden rahmen waere das relevant,
        # und dort faengt `prediction_sampling_max_segment_px` die luecke ab.
        d1x = (x1 - x0) - (2.0 * vx0 + vx1) * third
        d1y = (y1 - y0) - (2.0 * vy0 + vy1) * third
        d2x = (x0 - x1) + (vx0 + 2.0 * vx1) * third
        d2y = (y0 - y1) + (vy0 + 2.0 * vy1) * third
        scale = abs(float(camera.scale))
        second = np.maximum(np.hypot(d1x, d1y), np.hypot(d2x, d2y)) * scale
        second = np.where(np.isfinite(second) & candidate, second, 0.0)

        # Wie viele teilstuecke braucht es, damit der polygonzug hoechstens
        # `tol` von der kurve abweicht?
        #
        #   d <= max|B''| / (8 n^2)     und     max|B''| <= 6 M
        #   =>  d <= 0.75 M / n^2       =>  n = ceil(sqrt(0.75 M / tol))
        #
        # Der faktor ist nachgerechnet, nicht geraten: mit dem in vielen
        # rasterisierern kursierenden sqrt(3)/8 statt 3/4 wird um 1.86 zu
        # grob unterteilt (n = 4 statt 8), und genau das war messbar -- 1122 m
        # abweichung bei einer zusage von 1000 m, exakt sehne/n^2.
        tol = max(1e-4, float(eps_px))
        n_seg = np.ceil(np.sqrt(0.75 * second / tol))
        n_seg = np.where(np.isfinite(n_seg), n_seg, 1.0)
        counts = np.clip(n_seg - 1.0, 0.0,
                         float(max(0, int(self.prediction_hermite_max_subdiv)))
                         ).astype(np.int64)
        counts[~candidate] = 0

        total = int(counts.sum())
        if total <= 0:
            return None

        # BUDGET GLEICHMAESSIG DRUECKEN, NIE ABSCHNEIDEN. Wer das budget von
        # vorn aufbraucht, verliert das ende der linie -- und ein fehlender
        # horizont macht die anzeige unbrauchbar (Ap/Pe weg, CLOSEST leer),
        # waehrend eine gleichmaessig groebere linie nur etwas kantiger ist.
        room = int(budget) - int(idx.shape[0])
        if room < 0:
            room = 0
        if total > room:
            if room <= 0:
                return None
            factor = room / float(total)
            counts = np.floor(counts * factor).astype(np.int64)
            total = int(counts.sum())
            if total <= 0:
                return None
            stats['hermite_budget_limited'] = True

        # Auswertungsstellen: je segment `counts[i]` innere parameter.
        seg_id = np.repeat(np.arange(counts.shape[0], dtype=np.int64), counts)
        offsets = np.concatenate(([0], np.cumsum(counts)[:-1]))
        local = np.arange(total, dtype=np.int64) - offsets[seg_id]
        s = (local + 1).astype(np.float64) / (counts[seg_id] + 1).astype(np.float64)

        s2 = s * s
        s3 = s2 * s
        h00 = 2.0 * s3 - 3.0 * s2 + 1.0
        h10 = s3 - 2.0 * s2 + s
        h01 = -2.0 * s3 + 3.0 * s2
        h11 = s3 - s2

        seg_dt = dt[seg_id]
        ix = (h00 * x0[seg_id] + h10 * seg_dt * vx0[seg_id]
              + h01 * x1[seg_id] + h11 * seg_dt * vx1[seg_id])
        iy = (h00 * y0[seg_id] + h10 * seg_dt * vy0[seg_id]
              + h01 * y1[seg_id] + h11 * seg_dt * vy1[seg_id])
        it = t0[seg_id] + s * seg_dt

        # Grobe und neue punkte in EINE aufsteigende reihenfolge bringen.
        n_coarse = int(idx.shape[0])
        out = np.empty((n_coarse + total, 3), dtype=np.float64)
        before = np.concatenate(([0], np.cumsum(counts)))
        coarse_slots = np.arange(n_coarse, dtype=np.int64) + before
        out[coarse_slots, 0] = sub[:, 0]
        out[coarse_slots, 1] = sub[:, 1]
        out[coarse_slots, 2] = sub[:, 2]
        insert_slots = coarse_slots[seg_id] + 1 + local
        out[insert_slots, 0] = ix
        out[insert_slots, 1] = iy
        out[insert_slots, 2] = it

        stats['hermite_added'] = total
        stats['hermite_segments'] = int(np.count_nonzero(counts))
        return out

    def _runs_from_screen_points(self, screen_points, camera, tolerance_px,
                                 min_step_px, max_segment_px, max_points,
                                 margin_px, stats, coords=None):
        runs = self._build_clipped_polyline_runs(screen_points, margin_px,
                                                 coords=coords)
        stats['runs'] = len(runs)
        stats['clipped_runs'] = len(runs)
        if not runs:
            stats['draw_points'] = 0
            return []

        # ARRAYS STATT TUPEL-LISTEN. Verdichtung, RDP, luecken-auffuellung
        # und verdichtung-nach-innen sind alle indexoperationen auf denselben
        # zwei koordinaten-spalten. Frueher wanderte die spur nach jeder
        # stufe durch `list(zip(...tolist()))` und zurueck durch
        # `np.asarray` -- bei 4000 punkten je frame gemessen der zweit-
        # groesste einzelposten des zeichenwegs nach dem klippen selbst.
        if coords is not None:
            origin_x = float(coords[0][0])
            origin_y = float(coords[1][0])
        else:
            origin_x = float(screen_points[0][0])
            origin_y = float(screen_points[0][1])

        min_step2 = float(min_step_px) * float(min_step_px)
        tol2 = float(tolerance_px) * float(tolerance_px)
        max_seg = max(0.5, float(max_segment_px))

        sampled_runs = []

        for run in runs:
            run = np.asarray(run, dtype=np.float64)
            if run.ndim != 2 or run.shape[0] < 2:
                continue
            rxs = np.ascontiguousarray(run[:, 0])
            rys = np.ascontiguousarray(run[:, 1])

            run_starts_at_path_origin = (
                abs(rxs[0] - origin_x) < 1e-9 and
                abs(rys[0] - origin_y) < 1e-9
            )

            # Verdichtung: punkte naeher als min_step fallen weg.
            if _LINE_KERNELS_OK:
                cidx = _compact_min_step_numba(rxs, rys, min_step2)
            else:
                cidx = np.asarray(
                    self._compact_min_step_indices(rxs, rys, min_step2),
                    dtype=np.int64)
            cx = rxs[cidx]
            cy = rys[cidx]

            if cx.shape[0] > 2:
                if _LINE_KERNELS_OK:
                    keep_mask = _rdp_keep_numba(cx, cy, tol2)
                    keep_indices = np.nonzero(keep_mask)[0]
                else:
                    keep_indices = np.asarray(
                        self._rdp_indices(list(zip(cx.tolist(), cy.tolist())),
                                          tolerance_px),
                        dtype=np.int64)

                if run_starts_at_path_origin:
                    preserve_count = min(32, cx.shape[0])
                    # Die vereinigung mit 0..preserve_count-1 ist hier KEIN
                    # allgemeiner mengen-schnitt: der zweite operand ist ein
                    # LUECKENLOSER anfang, und `keep_indices` ist bereits
                    # sortiert und doppelfrei. Damit ist das ergebnis genau
                    # "0..preserve-1, dahinter alle keeps >= preserve" --
                    # eine suche plus ein anhaengen, ohne die hash-tabelle,
                    # die np.union1d aufbaut (gemessen 0.44 ms je aufruf,
                    # zwei aufrufe je frame).
                    cut = int(np.searchsorted(keep_indices, preserve_count,
                                              side='left'))
                    tail = keep_indices[cut:]
                    merged = np.empty(preserve_count + tail.shape[0],
                                      dtype=np.int64)
                    merged[:preserve_count] = np.arange(preserve_count,
                                                        dtype=np.int64)
                    merged[preserve_count:] = tail
                    keep_indices = merged

                # Guard against over-aggressive simplification by enforcing
                # a maximum screen-space gap between consecutive kept points.
                if keep_indices.shape[0] > 1:
                    keep_indices = np.ascontiguousarray(
                        keep_indices, dtype=np.int64)
                    if _LINE_KERNELS_OK:
                        keep_indices = _max_gap_refine_numba(
                            keep_indices, cx, cy, max_seg)
                    else:
                        keep_indices = self._max_gap_refine_indices(
                            keep_indices, cx, cy, max_seg)

                sampled_x = cx[keep_indices]
                sampled_y = cy[keep_indices]
            else:
                sampled_x = cx
                sampled_y = cy

            # Densify only the RDP-kept points, not the raw scan.
            # Pre-RDP densification of sparse predictors could expand 3000 samples
            # to 75 000+ linearly-interpolated dummies that RDP discards anyway,
            # making _rdp_indices O(N²) on a huge but information-free array.
            sampled = self._densify_screen_columns(
                sampled_x, sampled_y, max_segment_px)

            if sampled.shape[0] >= 2:
                sampled_runs.append(sampled)

        sampled_runs = self._cap_runs_by_screen_length(
            sampled_runs,
            self.prediction_render_max_screen_length_px,
            stats,
        )
        sampled_runs = self._cap_runs_by_point_budget(sampled_runs, max_points, stats)
        stats['drawn'] = sum(len(run) for run in sampled_runs)
        stats['draw_points'] = stats['drawn']
        stats['runs'] = len(sampled_runs)
        return sampled_runs

    def _draw_body_label(self, name, screen_pos, radius):
        # Label mit gecachten GL-Texturen zeichnen, um pro-Frame GL-Allocationen zu vermeiden.
        # Label horizontal zentrieren und über dem Körper platzieren, um
        # Fehlausrichtungen beim Zoomen oder bei Radiusänderungen zu vermeiden.
        try:
            entry = self._get_label_texture(name, self.font_small)
            if entry:
                _, w, h = entry
                label_x = float(screen_pos[0]) - (float(w) / 2.0)
                # screen_pos ist TOP-DOWN; ueber dem koerper heisst kleineres y.
                label_y = float(screen_pos[1]) - float(radius) - 6.0 - float(h)
                self._blit_text_topdown(name, label_x, label_y, self.font_small)
                return
        except Exception:
            pass

        # Fallback: previous heuristic
        label_x = screen_pos[0] + radius + 2
        label_y = screen_pos[1] - 8
        self._blit_text_topdown(name, label_x, label_y, self.font_small)
    
    def _draw_hud_quad(self, x, y, width, height):
        """Zeichnet die persistente HUD-textur als quad (ohne re-upload)."""
        if self._hud_texture is None:
            return
        self._draw_texture_ortho(self._hud_texture, x, y, width, height)

    def _blit_pygame_surface(self, surface, x, y):
        """Lädt eine pygame Surface in die persistente HUD-textur und zeichnet sie.

        Der upload (tostring + texture.write) ist der teure teil. Aufrufer,
        deren inhalt sich gegenüber dem vorframe nicht geändert hat, überspringen
        diese methode und rufen direkt _draw_hud_quad.
        """
        texture_data = pygame.image.tostring(surface, 'RGBA', True)
        width, height = surface.get_size()

        # Create or resize HUD texture
        if self._hud_texture is None or self._hud_texture_size != (width, height):
            if self._hud_texture is not None:
                try:
                    self._hud_texture.release()
                except Exception:
                    pass
            self._hud_texture = self.ctx.texture((width, height), 4, texture_data)
            self._hud_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._hud_texture_size = (width, height)
        else:
            self._hud_texture.write(texture_data)

        # Textur rendern
        self._draw_hud_quad(x, y, width, height)

    def _render_hud(self, camera, predictor=None):
        """Die alte debug-textwand unten links.

        Seit Phase 4 standardmaessig AUS: ihre werte stehen jetzt im
        spieler-HUD (spacesim/ui/hud/) bzw. im ImGui-entwicklerpanel (F1).
        Sie bleibt als schneller rohwert-blick erhalten und laesst sich ueber
        renderer.show_debug_hud in config.json wieder einschalten.
        """
        if not getattr(self, 'show_debug_hud', False):
            return

        # HUD-Texte vorbereiten
        def _fmt_dist(n):
            if n is None:
                return 'auto'
            try:
                n = float(n)
            except Exception:
                return str(n)
            if n >= 1e9:
                return f"{n/1e9:.2f}Gm"
            if n >= 1e6:
                return f"{n/1e6:.2f}Mm"
            if n >= 1e3:
                return f"{n/1e3:.2f}km"
            return f"{n:.0f}m"

        texts = [
            f"Scale: {camera.scale:.2e} px/m",
            f"Position: ({camera.position.x:.2e}, {camera.position.y:.2e})",
            f"Target: {camera.target.name if camera.target else 'None'}",
            f"Plot frame: {self._plotting_frame_label}",
            f"Target overlay: {self._target_frame_label if self._target_frame_label else 'OFF'}",
            f"Ref trails: {'ON' if self.reference_trajectories_enabled else 'OFF'}",
            f"Time step: {camera.sim_dt:.2e} s/step",
            f"Bodies rendered: {self.debug_info['bodies_rendered']}",
            f"FXAA: {'ON' if self.enable_fxaa else 'OFF'}",
        ]

        if predictor is not None:
            precision_factor = predictor.get_precision_factor() if hasattr(predictor, 'get_precision_factor') else 1.0
            display_length = predictor.get_display_length() if hasattr(predictor, 'get_display_length') else predictor.length
            texts += [
                f"Predictor len: {_fmt_dist(display_length)} ([+/-])",
                f"Predictor spacing: {_fmt_dist(predictor.precision)} ([9/0])",
                f"Predictor precision factor: {precision_factor:.2f}x",
                f"Pred points: {len(predictor.get_points())}/{predictor.num_points}",
                f"Pred draw points: {self.debug_info['prediction_points_drawn']}/{self.debug_info['prediction_points_in']}",
            ]
            if hasattr(predictor, 'get_async_status'):
                async_status = predictor.get_async_status()
                texts.append(
                    f"Pred async: {'ON' if async_status['enabled'] else 'OFF'} "
                    f"pending={async_status['pending']} swapped={async_status['swapped_jobs']}"
                )

        texts.append("[WASD] Move | [F] Unfollow | [Scroll] Zoom | [R] Cycle ref | [1]/[2] Frame mode | [T] Target overlay")
        
        # Pygame Surface für HUD erstellen. Alle maße sind design-einheiten und
        # werden über ui_px() auf die aktuelle fenstergröße skaliert; bei
        # ui_scale == 1.0 ergeben sich exakt die bisherigen festwerte.
        line_height = max(1, int(round(self.ui_px(16))))
        hud_width = max(1, int(round(self.ui_px(560))))
        margin = int(round(self.ui_px(10)))
        hud_height = max(
            int(round(self.ui_px(40))),
            len(texts) * line_height + int(round(self.ui_px(8))),
        )
        origin_x = margin
        origin_y = self.height - hud_height - margin

        # Bei unverändertem text bleibt die persistente HUD-textur gültig:
        # font.render (~1 pro zeile), Surface-allokation, tostring und der
        # textur-upload entfallen, es genügt ein redraw der bestehenden textur.
        cache_key = (tuple(texts), int(self.width), int(self.height), round(self.ui_scale, 3))
        if cache_key == self._hud_cache_key and self._hud_texture is not None:
            self._draw_hud_quad(origin_x, origin_y, *self._hud_texture_size)
            return

        hud_surface = pygame.Surface((hud_width, hud_height), pygame.SRCALPHA)

        # Zeilen-surfaces cachen: ändert sich nur eine zeile (z. B. kamera-
        # position beim verfolgen), durchlaufen die übrigen kein font.render.
        # Der cache wird pro frame auf die aktuellen texte reduziert und kann
        # daher nicht wachsen.
        new_line_cache = {}
        for i, text in enumerate(texts):
            text_surface = self._hud_line_surface_cache.get(text)
            if text_surface is None:
                text_surface = self.font_medium.render(text, True, (255, 255, 255))
            new_line_cache[text] = text_surface
            hud_surface.blit(text_surface, (0, i * line_height))
        self._hud_line_surface_cache = new_line_cache

        # HUD in OpenGL rendern
        self._blit_pygame_surface(hud_surface, origin_x, origin_y)
        self._hud_cache_key = cache_key
    
    def resize(self, width, height):

        self.width = width
        self.height = height
        self.ctx.viewport = (0, 0, width, height)
        # u_viewport haengt an der fenstergroesse -- der zustandscache waere
        # sonst genau ueber diesen wert veraltet.
        self._invalidate_gl_state_cache()
        # WICHTIG: moderngl erkennt die groesse von ctx.screen nur EINMAL beim
        # anlegen des contexts. Nach einem resize meldet ctx.screen.size noch
        # die alte fenstergroesse -- und jedes ctx.screen.use() stellt daraus
        # viewport UND scissor wieder her. Ohne die explizite neuzuweisung
        # unten klemmt der scissor nach dem FXAA-pass (render() ruft dort
        # ctx.screen.use()) alles nachfolgende -- predictor-linie, schiff, HUD
        # -- auf das alte fenster-rechteck: beim maximieren ist dann nur noch
        # ein ausschnitt des spiels sichtbar.
        # ctx.screen.scissor = None hilft NICHT: das setzt den scissor auf die
        # (weiterhin veraltete) eigengroesse des framebuffers zurueck.
        try:
            self.ctx.screen.viewport = (0, 0, width, height)
            self.ctx.screen.scissor = (0, 0, width, height)
        except Exception as exc:
            print(f"RENDERER WARNING: screen viewport/scissor resize failed ({exc})")

        # Framebuffer neu erstellen wenn FXAA aktiviert (programm + VAO
        # bleiben; nur textur/FBO hängen von der fenstergröße ab).
        if self.enable_fxaa:
            self._release_fxaa_targets()
            try:
                self._create_fxaa_targets()
                if self.fxaa_program is not None:
                    self.fxaa_program['u_resolution'].value = (float(width), float(height))
            except Exception as e:
                print(f"FXAA resize failed: {e}")
                self._release_fxaa_targets()
                self.enable_fxaa = False
        # UI-skala an die neue fensterhöhe anpassen. Nur bei echter änderung
        # die fonts neu aufbauen -- _rebuild_fonts leert die text-caches
        # ohnehin, und beim reinen breiten-resize wäre das verschwendet.
        if self._recompute_ui_scale():
            self._rebuild_fonts()

        # Clear HUD and label texture caches (will be recreated lazily)
        try:
            self._clear_text_caches()
        except Exception:
            pass
        if getattr(self, '_hud_texture', None):
            try:
                self._hud_texture.release()
            except Exception:
                pass
            self._hud_texture = None
            self._hud_texture_size = (0, 0)
        # HUD-memoization invalidieren: textur und viewport haben sich geändert.
        self._hud_cache_key = None
        # Der poly-VBO ist größen-unabhängig und bleibt (samt VAOs) bestehen.
