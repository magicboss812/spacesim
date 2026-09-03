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

from render import GL_DIR
from physics.reference_frames import IdentityReferenceFrame, apparent_orbital_directions
from render import background
from bodies import icon as body_icon
from bodies import style as body_style
from bodies import orbit_lines
from ship import art as ship_art
from render.background import BackgroundLayer

from runtime.gl_device import GLDeviceMixin
from render.pipelines import ShaderPipelineMixin
from render.draw import DrawMixin
from render.text import TextMixin
from render.background_draw import BackgroundDrawMixin
from render.bodies import BodyDrawMixin
from render.ship import ShipDrawMixin
from render.orbits import OrbitDrawMixin
from render.prediction import PredictionDrawMixin
from render.line_kernels import (
    _LINE_KERNELS_OK,
    _clip_runs_numba,
    _compact_min_step_numba,
    _densify_numba,
    _max_gap_refine_numba,
    _rdp_keep_numba,
)


class Renderer(
    GLDeviceMixin,
    ShaderPipelineMixin,
    DrawMixin,
    TextMixin,
    BackgroundDrawMixin,
    BodyDrawMixin,
    ShipDrawMixin,
    OrbitDrawMixin,
    PredictionDrawMixin,
):
    """Der Renderer -- zusammengesetzt aus mixins, ein zustand.

    Die klasse war 5900 zeilen in EINER datei. Sie ist jetzt ueber `render/`
    verteilt, aber weiterhin EIN objekt: die mixins teilen sich `self` und die
    dutzenden felder, die `__init__` unten anlegt. Das ist bewusst so und nicht
    als komposition (`self.gl.line_program` statt `self._line_program`)
    umgesetzt -- letzteres haette hunderte attributzugriffe quer durch die
    groesste datei des projekts umgeschrieben, fuer eine rein strukturelle
    aenderung ein schlechtes verhaeltnis von risiko zu ertrag.

    Was hier BLEIBT, ist der kern, an dem alles haengt: der zustand
    (`__init__`), der frame-durchlauf (`render`), die projektion in den aktiven
    plot-rahmen und das HUD.
    """
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
        # DER KOERPERNAME LAEUFT UEBER DIE HAUSSCHRIFT, NICHT UEBER DIE
        # SYSTEMSCHRIFT. Er ist die einzige beschriftung, die MITTEN im bild
        # steht -- neben einem HUD, das durchgehend SB Liquid setzt, fiel
        # ausgerechnet der name des ausgewaehlten koerpers als fremde
        # groteske heraus. Gesetzt wird er wie jede display-beschriftung der
        # oberflaeche: VERSAL, gesperrt, hart gerastert und auf ein
        # vielfaches von fuenf pixel gerundet (siehe ui/theme.py, modulkopf
        # und .claude/rules/ui-hud.md).
        self.hud_font_size_body_label = 15
        self.body_label_uppercase = True
        self.body_label_tracking_em = 0.12
        self.font_body_label = None
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
        # DIE MARKER MELDEN IHRE SCHIRMPOSITION, ZEICHNEN IHREN SCHWEBEZETTEL
        # ABER NICHT SELBST. Der zettel ist ein HUD-element (ui/hud/
        # apsis_tooltip.py): er braucht die hausschrift, den SDF-shader fuer
        # seine flaeche und die maus -- alles drei liegt in ui/, nicht hier.
        # Der renderer ist die einzige stelle, die die frame-abhaengige
        # transformation der marker kennt, also legt er das ergebnis hier ab
        # und das HUD liest es. Eine zeile je marker, real ein bis vier.
        self.apsis_tooltip_enabled = True
        self.apsis_tooltip_hover_px = 14.0
        #: (sx, sy, radius_px, is_apoapsis, distance_m, t_abs, alpha)
        self.apsis_marker_hits = []
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
        # Winkel-boden der zukunfts-spur (OrbitLineSet.__init__): mindestens
        # so viele stuetzstellen je umlauf, hoechstens so viele insgesamt,
        # und hoechstens so viele umlaeufe ueberhaupt gezeichnet.
        self.orbit_line_samples_per_period = 64.0
        self.orbit_line_max_track_samples = 1024
        self.orbit_line_max_periods_drawn = 16.0
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
        self._shader_dir = GL_DIR

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





























    #: Wie weit das marken-quad ueber den radius hinausreicht (fuer den halo).
    ICON_QUAD_EXTENT = 2.6


















    #: Rasterstufe der pixelschrift und ihre kleinste brauchbare groesse --
    #: dieselben zahlen wie in ui/text.py::_role_pixel_size, und aus demselben
    #: grund: SB Liquid ist auf einem pixelraster gezeichnet, und nur auf
    #: vielfachen von fuenf bleiben ihre stege ueberall gleich breit.
    _DISPLAY_PIXEL_STEP = 5
    _DISPLAY_PIXEL_MIN = 10





    # Deckel des texturen-recyclings, siehe _acquire_label_texture.
    _LABEL_TEXTURE_POOL_MAX = 64
















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
            timings['bodies_ms'] += (time.perf_counter() - bodies_t0) * 1000.0

        # Auswahl-markierung ebenfalls nach dem FXAA-resolve, aus demselben
        # grund wie die beschriftungen -- und vor ihnen, damit ein label nicht
        # unter einem pfeil verschwindet.
        self._draw_selection_marker(camera)

        # Körper-beschriftungen erst jetzt zeichnen -- nach dem FXAA-resolve,
        # damit der kantenfilter den text nicht verschmiert (siehe _draw_body).
        for name, label_x, label_y, font, antialias, tracking in self._deferred_labels:
            self._blit_text_topdown(name, label_x, label_y, font,
                                    antialias=antialias, tracking=tracking)

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
    
        # Der poly-VBO ist größen-unabhängig und bleibt (samt VAOs) bestehen.
