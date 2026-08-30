import json
import re
from pathlib import Path
from typing import Any
from vec import Vec2
from bodies import body, schiff
import background

DEFAULT_CONFIG_FILE = "config.json"


def _float_list(value):
    """JSON-liste -> aufsteigend sortierte liste positiver floats.

    Wird fuer `renderer.prediction_error_ladder_m` gebraucht. Sortiert und
    filtert hier, damit die auswahl der sprosse sich darauf verlassen kann --
    eine unsortierte leiter waehlt sonst still die falsche.
    """
    if isinstance(value, (int, float)):
        value = [value]
    rungs = sorted(float(v) for v in value if float(v) > 0.0)
    if not rungs:
        raise ValueError("leere toleranz-leiter")
    return rungs


def _strip_jsonc(text):
    """Entfernt kommentare und abschliessende kommata aus einem JSON-text.

    `config.json` ist reines, striktes JSON — diese funktion ist reine
    nachsicht beim einlesen: wer beim editieren eine `//`-notiz oder ein
    ueberzaehliges komma stehen laesst, bekommt keinen parse-fehler um die
    ohren. `//` innerhalb von strings (z. B. in pfaden) bleibt erhalten,
    deshalb wird zeichenweise geparst statt per regex.
    """
    out = []
    in_string = False
    escaped = False
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if in_string:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == '\\':
                escaped = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue
        if ch == '/' and i + 1 < n and text[i + 1] == '/':
            while i < n and text[i] != '\n':
                i += 1
            continue
        if ch == '/' and i + 1 < n and text[i + 1] == '*':
            i += 2
            while i + 1 < n and not (text[i] == '*' and text[i + 1] == '/'):
                i += 1
            i += 2
            continue
        out.append(ch)
        i += 1

    stripped = ''.join(out)
    # abschliessende kommata tolerieren, damit das um- und auskommentieren
    # einzelner zeilen die datei nicht sofort ungueltig macht.
    return re.sub(r',(\s*[}\]])', r'\1', stripped)


class ConfigLoader:
    """Laedt `config.json` und verteilt die parameter auf die einzelnen module.

    die datei ist die einzige stelle, an der spielbare parameter stehen; diese
    klasse traegt sie an die jeweiligen objekte (world, Camera, schiffcontrol,
    Predictor, Renderer) und an die modulglobalen konstanten (G) heran. jeder
    `apply_*`-aufruf setzt nur die schluessel, die auch wirklich in der datei
    stehen — fehlende schluessel behalten den default im code.
    """

    def __init__(self, filepath: str | Path | None = None):
        base_dir = Path(__file__).parent
        if filepath is None:
            self.filepath = base_dir / DEFAULT_CONFIG_FILE
        else:
            p = Path(filepath)
            if not p.is_absolute():
                p = base_dir / p
            self.filepath = p
        self.data: dict[str, Any] = {}
        self.unknown_keys: list[str] = []

    # ---------------------------------------------------------------- laden

    def load(self) -> dict[str, Any]:
        """Liest die konfiguration ein und gibt sie zurueck.

        `utf-8-sig` statt `utf-8`: windows-editoren speichern gern mit BOM, die
        der JSON-parser sonst als ungueltiges zeichen ablehnt.
        """
        with open(self.filepath, 'r', encoding='utf-8-sig') as f:
            raw = f.read()
        try:
            data = json.loads(_strip_jsonc(raw))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{self.filepath.name}: ungueltige konfiguration ({exc})") from exc
        if not isinstance(data, dict):
            raise ValueError(f"{self.filepath.name}: erwartet ein JSON-objekt auf oberster ebene")
        self.data = data
        return self.data

    def section(self, name: str) -> dict[str, Any]:
        """Gibt einen abschnitt (z. B. 'camera') als dict zurueck, notfalls leer."""
        value = self.data.get(name, {})
        return value if isinstance(value, dict) else {}

    def get(self, dotted_path: str, default: Any = None) -> Any:
        """Liest einen wert ueber einen punkt-pfad, z. B. get('camera.zoom_factor').

        rueckgabetyp ist bewusst `Any`: eine konfigurationsdatei liefert je nach
        schluessel zahl, text, wahrheitswert oder abschnitt. wer einen bestimmten
        typ braucht, nimmt `get_float` / `get_int` / `get_bool` / `get_str` —
        die pruefen den wert, statt ihn blind an `float()` weiterzureichen.
        """
        node: Any = self.data
        for part in str(dotted_path).split('.'):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    # -------------------------------------------------------- typsichere gets

    def _coerce(self, where: str, value: Any, caster, default: Any) -> Any:
        """Wandelt `value` mit `caster` um; bei unpassendem wert warnung + default.

        so fuehrt ein tippfehler in der konfiguration (z. B. `"width": "gross"`
        oder ein versehentlich verschachtelter abschnitt) zu einer verstaendlichen
        meldung und dem eingebauten default, statt zu einem absturz beim start.
        """
        if value is None:
            return default
        try:
            return caster(value)
        except (TypeError, ValueError) as exc:
            print(f"CONFIG WARNING: {where}={value!r} ungueltig ({exc}), verwende {default!r}")
            return default

    def get_float(self, dotted_path: str, default: float | None = None) -> Any:
        """Liest einen kommazahl-parameter."""
        return self._coerce(dotted_path, self.get(dotted_path), float, default)

    def get_int(self, dotted_path: str, default: int | None = None) -> Any:
        """Liest einen ganzzahl-parameter."""
        return self._coerce(dotted_path, self.get(dotted_path), int, default)

    def get_str(self, dotted_path: str, default: str | None = None) -> Any:
        """Liest einen text-parameter."""
        return self._coerce(dotted_path, self.get(dotted_path), str, default)

    def get_bool(self, dotted_path: str, default: bool | None = None) -> Any:
        """Liest einen an/aus-parameter.

        akzeptiert auch die schreibweisen "false"/"0"/"no"/"off" als aus, damit
        ein in anfuehrungszeichen gesetzter wert nicht stillschweigend zu `True`
        wird (`bool("false")` ist in python wahr).
        """
        value = self.get(dotted_path)
        if value is None:
            return default
        if isinstance(value, str):
            text = value.strip().lower()
            if text in ("0", "false", "no", "off", ""):
                return False
            if text in ("1", "true", "yes", "on"):
                return True
            print(f"CONFIG WARNING: {dotted_path}={value!r} ungueltig, verwende {default!r}")
            return default
        return self._coerce(dotted_path, value, bool, default)

    # ------------------------------------------------------------ zuweisung

    def _assign(self, target, section_name, spec):
        """Setzt attribute auf `target` gemaess `spec` = [(json_key, attr, cast)]."""
        section = self.section(section_name)
        used = set()
        for key, attr, cast in spec:
            if key not in section:
                continue
            used.add(key)
            if attr is None:
                # schluessel wird anderswo ausgewertet (z. B. direkt in test.py);
                # hier nur als "bekannt" markieren, damit keine warnung kommt.
                continue
            value = section[key]
            try:
                setattr(target, attr, cast(value) if cast is not None else value)
            except (TypeError, ValueError) as exc:
                print(f"CONFIG WARNING: {section_name}.{key}={value!r} ignoriert ({exc})")
        for key in section:
            if key not in used:
                self.unknown_keys.append(f"{section_name}.{key}")
        return target

    def apply_globals(self):
        """Traegt die Gravitationskonstante in alle module ein, die sie halten.

        `G` wird in mehreren modulen per `from vec import G` gebunden; ein
        aendern von `vec.G` allein wuerde die bereits importierten kopien nicht
        erreichen, deshalb werden sie hier einzeln gesetzt.
        """
        g = self.get_float('physics.gravitational_constant')
        if g is None:
            return None
        import vec as _vec
        import bodies as _bodies
        import world as _world
        _vec.G = g
        _bodies.G = g
        _world.G = g
        try:
            import reference_frames as _rf
            _rf.NEWTONIAN_G = g
        except Exception:
            pass
        return g

    def apply_to_world(self, world_obj):
        """physics-abschnitt -> world.py (integrator und Gravitationskonstante)."""
        self._assign(world_obj, 'physics', [
            ('gravitational_constant', 'G', float),
            ('integrator_mode', 'integrator_mode', str),
            ('integrator_max_step', 'integrator_max_step', float),
            ('integrator_warp_substep_target', 'integrator_warp_substep_target', float),
            ('integrator_max_step_ceiling', 'integrator_max_step_ceiling', float),
            ('integrator_min_step', 'integrator_min_step', float),
            ('integrator_position_tolerance', 'integrator_position_tolerance', float),
            ('integrator_velocity_tolerance', 'integrator_velocity_tolerance', float),
        ])
        debug = self.get_bool('debug.world_integrator_debug')
        if debug is not None:
            world_obj.integrator_debug = debug
        return world_obj

    def apply_to_camera(self, camera):
        """camera- und simulation-abschnitt -> camera.py (zoom, schwenken, zeitraffer)."""
        self._assign(camera, 'camera', [
            ('initial_scale', 'scale', float),
            ('min_scale', 'min_scale', float),
            ('max_scale', 'max_scale', float),
            ('min_visible_span_m', 'min_visible_span_m', float),
            ('move_speed', 'move_speed', float),
            ('zoom_factor', 'zoom_factor', float),
            ('zoom_smoothing', 'zoom_smoothing', float),
            ('pan_smoothing', 'pan_smoothing', float),
            ('focus_smoothing', 'focus_smoothing', float),
            ('pan_inertia_enabled', 'pan_inertia_enabled', bool),
            ('pan_inertia_damping', 'pan_inertia_damping', float),
        ])
        # initial_scale setzt die GEZEICHNETE skala; das zoom-ziel muss
        # mitgezogen werden, sonst laeuft die kamera beim start sofort von
        # der konfigurierten skala auf den (noch unveraenderten) default zurueck.
        try:
            camera.target_scale = camera.scale
            camera.snap_to_targets()
        except Exception:
            pass
        sim = self.section('simulation')
        for key, attr in (('initial_sim_dt', 'sim_dt'),
                          ('min_sim_dt', 'min_sim_dt'),
                          ('max_sim_dt', 'max_sim_dt'),
                          ('sim_dt_step_factor', 'sim_dt_factor')):
            if key in sim:
                value = self._coerce(f'simulation.{key}', sim[key], float, getattr(camera, attr))
                setattr(camera, attr, value)
        return camera

    def apply_to_ship_control(self, ship_control):
        """ship-abschnitt -> schiff.py (drehrate und schubstaerke)."""
        if ship_control is None:
            return None
        return self._assign(ship_control, 'ship', [
            ('rotation_speed', 'rotation_speed', float),
            ('thrust_acc', 'thrust_acc', float),
        ])

    def predictor_kwargs(self):
        """Konstruktor-argumente fuer Predictor (was nach dem bau nicht mehr geht).

        `async_compute` / `rolling_mode` steuern in `Predictor.__init__` den
        aufbau des worker-executors und werden deshalb schon dort uebergeben.
        """
        p = self.section('predictor')
        kwargs: dict[str, Any] = {}
        for key, getter in (('num_points', self.get_int),
                            ('dt', self.get_float),
                            ('precision', self.get_float),
                            ('async_compute', self.get_bool),
                            ('rolling_mode', self.get_bool),
                            ('strict_snapshot_matching', self.get_bool),
                            ('use_time_dependent_bodies', self.get_bool)):
            if key in p:
                value = getter(f'predictor.{key}')
                # bei ungueltigem wert liefert der getter None -> konstruktor-
                # default des Predictors greift, statt ihn mit None zu fuettern.
                if value is not None:
                    kwargs[key] = value
        kwargs['debug'] = bool(self.get_bool('debug.predictor_debug', False))
        return kwargs

    def apply_to_predictor(self, predictor):
        """predictor-abschnitt -> predictor.py (genauigkeit, reichweite, marker)."""
        if predictor is None:
            return None
        quality = self.get_str('predictor.quality')
        if quality is not None:
            try:
                predictor.set_integrator_quality(quality)
                # Das preset merken: set_integrator_quality verteilt es auf
                # einzelne toleranzen und laesst sich daraus nicht mehr
                # rueckwaerts ablesen. Die dev-oberflaeche zeigt es an.
                predictor._quality = str(quality).strip().lower()
            except ValueError as exc:
                print(f"CONFIG WARNING: predictor.quality={quality!r} ignoriert ({exc})")

        self._assign(predictor, 'predictor', [
            # bereits im konstruktor gesetzt (siehe predictor_kwargs), hier nur
            # der vollstaendigkeit halber erneut, damit die datei fuehrend bleibt
            ('num_points', 'num_points', int),
            ('dt', 'dt', float),
            ('precision', 'precision', float),
            # async_compute/rolling_mode nur im konstruktor (executor-aufbau),
            # hier bewusst nicht nachtraeglich ueberschreiben
            ('async_compute', None, None),
            ('rolling_mode', None, None),
            ('strict_snapshot_matching', 'strict_snapshot_matching', bool),
            ('use_time_dependent_bodies', 'use_time_dependent_bodies', bool),
            # laufzeit-regler
            ('min_precision', 'min_precision', float),
            ('auto_precision_from_zoom', 'auto_precision_from_zoom', bool),
            ('target_screen_step_px', 'target_screen_step_px', float),
            ('force_sync_on_stale', 'force_sync_on_stale', bool),
            ('use_reference_acceleration_correction', 'use_reference_acceleration_correction', bool),
            ('rkn_adaptive_far_maxdt', 'rkn_adaptive_far_maxdt', bool),
            ('rkn_far_field_target_steps', 'rkn_far_field_target_steps', float),
            ('rkn_max_dt_ceiling', 'rkn_max_dt_ceiling', float),
            ('apsis_markers_enabled', 'apsis_markers_enabled', bool),
            ('apsis_max_markers', 'apsis_max_markers', int),
            # Gleichzeitige vorhersagen unter schub. Eine rechnung dauert
            # laenger als ein bild, also gibt erst der DURCHSATZ mehrerer
            # zeitversetzter laeufe eine bildweise nachziehende linie.
            # 1 = wie vorher (eine nach der anderen).
            ('thrust_pipeline_depth', 'thrust_pipeline_depth', int),
            # Jitter-puffer fuer das einwechseln fertiger vorhersagen.
            # 0 = immer sofort das neueste (kuerzeste verzoegerung, aber
            # ruckartig), 2 = vollstaendig gleichmaessig, dafuer aeltere linie.
            ('swap_backlog_max', 'swap_backlog_max', int),
            # nur von test.py ausgewertet (tastenbelegung / ein-aus-verhalten)
            ('quality', None, None),
            ('enabled', None, None),
            ('toggle_num_points', None, None),
            ('length_step_factor', None, None),
            ('precision_step_factor', None, None),
            ('max_num_points', None, None),
        ])
        # `precision` ist der basiswert, von dem die zoom-automatik ausgeht.
        if 'precision' in self.section('predictor'):
            predictor.base_precision = float(predictor.precision)

        debug_sources = self.get_bool('debug.predictor_debug_moving_sources')
        if debug_sources is not None:
            predictor.debug_moving_sources = debug_sources
        return predictor

    def apply_to_renderer(self, renderer):
        """renderer-abschnitt -> rendering.py (linienqualitaet, marker, HUD)."""
        if renderer is None:
            return None
        self._assign(renderer, 'renderer', [
            ('prediction_sampling_tolerance_px', 'prediction_sampling_tolerance_px', float),
            ('prediction_sampling_min_tolerance_px', 'prediction_sampling_min_tolerance_px', float),
            ('prediction_sampling_max_tolerance_px', 'prediction_sampling_max_tolerance_px', float),
            ('prediction_sampling_min_step_px', 'prediction_sampling_min_step_px', float),
            ('prediction_sampling_max_segment_px', 'prediction_sampling_max_segment_px', float),
            ('prediction_sampling_max_points', 'prediction_sampling_max_points', int),
            ('prediction_sampling_reference_scale', 'prediction_sampling_reference_scale', float),
            ('prediction_visibility_margin_px', 'prediction_visibility_margin_px', float),
            ('prediction_render_max_raw_scan', 'prediction_render_max_raw_scan', int),
            ('prediction_render_max_draw_points', 'prediction_render_max_draw_points', int),
            ('prediction_bypass_fxaa', 'prediction_bypass_fxaa', bool),
            # Aufloesungsgetriebene verfeinerung der vorhersagelinie.
            ('prediction_hermite_enabled', 'prediction_hermite_enabled', bool),
            ('prediction_detail_scale', 'prediction_detail_scale', float),
            ('prediction_hermite_max_subdiv', 'prediction_hermite_max_subdiv', int),
            ('prediction_error_ladder_m', 'prediction_error_ladder_m', _float_list),
            # Das spieler-HUD haengt nicht am renderer, sondern wird in
            # test.py aufgebaut -- hier nur als bekannt markieren.
            ('hud_enabled', None, None),
            ('show_debug_hud', 'show_debug_hud', bool),
            ('show_apsis_markers', 'show_apsis_markers', bool),
            ('apsis_marker_radius_px', 'apsis_marker_radius_px', float),
            ('apsis_marker_fade_min_px', 'apsis_marker_fade_min_px', float),
            ('apsis_marker_fade_full_px', 'apsis_marker_fade_full_px', float),
            ('body_icon_min_radius_px', 'body_icon_min_radius_px', float),
            # Die positions-marke (body_icon.py)
            ('body_icon_style', 'body_icon_style', str),
            ('body_icon_variant', 'body_icon_variant', str),
            ('body_icon_seed_offset', 'body_icon_seed_offset', int),
            ('body_icon_grid', 'body_icon_grid', int),
            # Skalierung der marken-groesse mit dem echten koerper-radius
            ('body_icon_max_radius_px', 'body_icon_max_radius_px', float),
            ('body_icon_size_influence', 'body_icon_size_influence', float),
            ('body_icon_fade_factor', 'body_icon_fade_factor', float),
            ('body_icon_halo_alpha', 'body_icon_halo_alpha', float),
            ('body_icon_edge_px', 'body_icon_edge_px', float),
            ('body_icon_cell_gap', 'body_icon_cell_gap', float),
            ('body_icon_shade_jitter', 'body_icon_shade_jitter', float),
            ('body_icon_cell_rim', 'body_icon_cell_rim', float),
            ('body_icon_cell_rim_dark', 'body_icon_cell_rim_dark', float),
            # Wann ein koerper seinen namen zeigt: "selected" | "zoom" | "both"
            ('body_label_mode', 'body_label_mode', str),
            ('body_label_min_radius_px', 'body_label_min_radius_px', float),
            # Schiffs-grafik (ship_art.py)
            ('ship_sprite_enabled', 'ship_sprite_enabled', bool),
            ('ship_length_px', 'ship_length_px', float),
            ('ship_render_scale', 'ship_render_scale', float),
            ('ship_accent_color', 'ship_accent_color', str),
            ('ship_plume_idle', 'ship_plume_idle', float),
            # Zoom-abhaengige verkleinerung des schiffs (px je meter)
            ('ship_zoom_shrink_enabled', 'ship_zoom_shrink_enabled', bool),
            ('ship_zoom_shrink_start_scale', 'ship_zoom_shrink_start_scale', float),
            ('ship_zoom_shrink_end_scale', 'ship_zoom_shrink_end_scale', float),
            ('ship_zoom_shrink_min', 'ship_zoom_shrink_min', float),
            # Auswahl-markierung (vier pfeile um den angeklickten koerper)
            ('selection_marker_enabled', 'selection_marker_enabled', bool),
            ('selection_arrow_length_px', 'selection_arrow_length_px', float),
            ('selection_arrow_width_px', 'selection_arrow_width_px', float),
            ('selection_gap_px', 'selection_gap_px', float),
            ('selection_min_radius_px', 'selection_min_radius_px', float),
            ('selection_max_radius_px', 'selection_max_radius_px', float),
            ('selection_spin_deg_per_s', 'selection_spin_deg_per_s', float),
            ('selection_pulse_period_s', 'selection_pulse_period_s', float),
            ('selection_pulse_amount', 'selection_pulse_amount', float),
            ('selection_pick_margin_px', 'selection_pick_margin_px', float),
            # Prozedurale vektor-optik der koerper (D2)
            ('body_vector_style', 'body_vector_style', bool),
            ('body_vector_min_radius_px', 'body_vector_min_radius_px', float),
            ('body_vector_full_radius_px', 'body_vector_full_radius_px', float),
            ('body_vector_detail', 'body_vector_detail', None),
            ('body_vector_facet_px', 'body_vector_facet_px', float),
            ('body_vector_detail_blend', 'body_vector_detail_blend', float),
            ('body_vector_coverage', 'body_vector_coverage', float),
            ('body_vector_shape_density', 'body_vector_shape_density', float),
            ('body_light_enabled', 'body_light_enabled', bool),
            ('body_ambient', 'body_ambient', float),
            ('body_light_exponent', 'body_light_exponent', float),
            ('body_light_tilt', 'body_light_tilt', float),
            ('body_glow_alpha', 'body_glow_alpha', float),
            # Bahnlinien der koerper (orbit_lines.py)
            ('orbit_lines_enabled', 'orbit_lines_enabled', bool),
            ('orbit_line_tolerance_px', 'orbit_line_tolerance_px', float),
            ('orbit_line_min_screen_px', 'orbit_line_min_screen_px', float),
            ('orbit_line_track_samples', 'orbit_line_track_samples', int),
            ('orbit_line_soi_full', 'orbit_line_soi_full', float),
            ('orbit_line_soi_fade', 'orbit_line_soi_fade', float),
            ('orbit_line_reveal_full', 'orbit_line_reveal_full', float),
            ('orbit_line_reveal_fade', 'orbit_line_reveal_fade', float),
            ('orbit_line_alpha_max', 'orbit_line_alpha_max', float),
            ('orbit_line_alpha_floor', 'orbit_line_alpha_floor', float),
            ('orbit_line_alpha_floor_focus', 'orbit_line_alpha_floor_focus', float),
            ('orbit_line_fade_rate', 'orbit_line_fade_rate', float),
            ('orbit_line_width', 'orbit_line_width', float),
            ('orbit_line_knot_angle', 'orbit_line_knot_angle', float),
            ('orbit_line_end_caps', 'orbit_line_end_caps', bool),
            ('orbit_line_end_cap_px', 'orbit_line_end_cap_px', float),
            # Faint volllinie: ein ganzer umlauf, hinter der enthuellten spur
            ('orbit_line_full_orbit_enabled', 'orbit_line_full_orbit_enabled', bool),
            ('orbit_line_full_alpha_mult', 'orbit_line_full_alpha_mult', float),
            ('orbit_line_full_knot_angle', 'orbit_line_full_knot_angle', float),
            ('orbit_line_full_samples', 'orbit_line_full_samples', int),
            ('orbit_line_full_max_span_s', 'orbit_line_full_max_span_s', float),
            ('reference_trajectories_enabled', 'reference_trajectories_enabled', bool),
            ('reference_trajectories_max_points', 'reference_trajectories_max_points', int),
            ('reference_trajectories_sample_step_s', 'reference_trajectories_sample_step_s', float),
            ('reference_traj_min_screen_px', 'reference_traj_min_screen_px', float),
            ('label_texture_cache_max', '_label_texture_cache_max', int),
            # UI-skalierung (aufloesungsunabhaengige darstellung)
            ('ui_scale_reference_height', 'ui_scale_reference_height', float),
            ('ui_scale_min', 'ui_scale_min', float),
            ('ui_scale_max', 'ui_scale_max', float),
            # schriftgroessen werden unten ueber set_hud_font_sizes gesetzt
            ('hud_font_size_small', None, None),
            ('hud_font_size_medium', None, None),
            ('ui_scale', None, None),
        ])

        section = self.section('renderer')
        font_small = self.get_int('renderer.hud_font_size_small') if 'hud_font_size_small' in section else None
        font_medium = self.get_int('renderer.hud_font_size_medium') if 'hud_font_size_medium' in section else None

        # Benutzer-skalenfaktor zuerst: er geht in die font-pixelgroesse ein,
        # die set_hud_font_sizes() gleich darunter berechnet.
        ui_scale_user = self.get_float('renderer.ui_scale') if 'ui_scale' in section else None
        try:
            if ui_scale_user is not None:
                renderer.set_ui_scale_user(ui_scale_user)
            else:
                # ui_scale_reference_height/min/max koennen sich oben geaendert
                # haben -> skala neu ableiten.
                renderer.set_ui_scale_user(getattr(renderer, 'ui_scale_user', 1.0))
        except Exception as exc:
            print(f"CONFIG WARNING: UI-skalierung konnte nicht gesetzt werden ({exc})")

        if font_small is not None or font_medium is not None:
            try:
                # set_hud_font_sizes speichert DESIGN-groessen und erzeugt die
                # fonts in groesse * ui_scale neu; der label-textur-cache wird
                # dabei geleert (er ist nach schrifthoehe verschluesselt).
                renderer.set_hud_font_sizes(small=font_small, medium=font_medium)
            except Exception as exc:
                print(f"CONFIG WARNING: HUD-schriftgroesse konnte nicht gesetzt werden ({exc})")

        for key, attr in (('renderer_debug_predictor', 'debug_predictor'),
                          ('renderer_debug_frame', 'debug_frame'),
                          ('render_benchmark_debug', 'render_benchmark_debug')):
            value = self.get_bool(f'debug.{key}')
            if value is not None:
                setattr(renderer, attr, value)
        every_n = self.get_int('debug.render_benchmark_every_n_frames')
        if every_n is not None:
            renderer.render_benchmark_every_n_frames = every_n

        # Die hintergrund-ebene haengt am renderer, hat aber einen eigenen
        # config-abschnitt -- hier mitgereicht, damit apply_all() nichts
        # zusaetzliches uebergeben muss.
        self.apply_to_background(getattr(renderer, 'background', None))
        return renderer

    def apply_to_background(self, bg):
        """background-abschnitt -> background.BackgroundLayer.

        Die schluesselmenge hier, die im `background`-abschnitt von
        config.json und die reglerliste im ImGui-panel sind ABSICHTLICH
        identisch -- so kann keiner der drei orte einen schalter bekommen,
        den die anderen nicht kennen (siehe .claude/rules/background.md).
        """
        if bg is None:
            return None
        self._assign(bg, 'background', [
            ('enabled', 'enabled', bool),
            ('grid_enabled', 'grid_enabled', bool),
            ('stars_enabled', 'stars_enabled', bool),
            ('accent_color', 'accent_color', str),
            ('grid_opacity', 'grid_opacity', float),
            ('grid_anchor', 'grid_anchor', str),
            ('idle_fade_delay', 'idle_fade_delay', float),
            ('pixel_size', 'pixel_size', float),
            ('pixel_round', 'pixel_round', float),
            ('grid_max_speed_px', 'grid_max_speed_px', float),
            ('star_density', 'star_density', int),
            ('star_opacity', 'star_opacity', float),
            ('star_motion_scale', 'star_motion_scale', float),
            ('star_zoom_influence', 'star_zoom_influence', float),
        ])
        # Ein tippfehler im anker darf nicht still ein anderes verhalten
        # ergeben -- er faellt hoerbar auf die Vorgabe zurueck.
        if bg.grid_anchor not in background.GRID_ANCHORS:
            print(f"CONFIG WARNING: background.grid_anchor={bg.grid_anchor!r} "
                  f"unbekannt, verwende {background.GRID_ANCHORS[0]!r} "
                  f"(erlaubt: {', '.join(background.GRID_ANCHORS)})")
            bg.grid_anchor = background.GRID_ANCHORS[0]
        return bg

    def apply_all(self, world_obj=None, camera=None, ship_control=None,
                  predictor=None, renderer=None):
        """Bequemlichkeit: alle abschnitte auf einmal verteilen."""
        self.apply_globals()
        if world_obj is not None:
            self.apply_to_world(world_obj)
        if camera is not None:
            self.apply_to_camera(camera)
        if ship_control is not None:
            self.apply_to_ship_control(ship_control)
        if predictor is not None:
            self.apply_to_predictor(predictor)
        if renderer is not None:
            self.apply_to_renderer(renderer)
        if self.unknown_keys and self.get_bool('debug.print_loader_info', True):
            print(f"CONFIG: unbekannte schluessel ignoriert: {', '.join(sorted(set(self.unknown_keys)))}")
        return self


class SystemLoader:
    """Lädt ein Planetensystem aus einer JSON-Datei."""
    
    def __init__(self, filepath=None):
        """Initialisiert den Loader.

        falls `filepath` weggelassen oder relativ angegeben wird, wird es
        relativ zum verzeichnis dieses moduls aufgelöst, sodass der loader
        unabhängig vom aktuellen arbeitsverzeichnis funktioniert.
        """
        base_dir = Path(__file__).parent
        if filepath is None:
            self.filepath = base_dir / "solar_system.json"
        else:
            p = Path(filepath)
            if not p.is_absolute():
                p = base_dir / p
            self.filepath = p
        self.data = None
    
    def load(self):
        """Lädt die JSON-Datei und gibt eine Liste von body-Objekten zurück."""
        # utf-8-sig: toleriert eine BOM, wie sie windows-editoren schreiben.
        with open(self.filepath, 'r', encoding='utf-8-sig') as f:
            self.data = json.load(f)
        
        bodies = []
        body_refs = {}  # Für Parent-Referenzen (Monde)
        
        # Erster Durchlauf: Alle Körper erstellen
        for entry in self.data.get("bodies", []):
            b = self._create_body(entry)
            bodies.append(b)
            body_refs[entry["name"]] = b
        
        # Zweiter Durchlauf: Parent-Referenzen auflösen (is_moon_of)
        for i, entry in enumerate(self.data.get("bodies", [])):
            if "is_moon_of" in entry and entry["is_moon_of"]:
                bodies[i].is_moon_of = body_refs.get(entry["is_moon_of"])
                bodies[i].scripted_orbit = True

        return bodies
    
    def hex_to_rgb(self, hex_color):
        """Wandelt Hex-String '#RRGGBB' in RGB-Tupel (r, g, b) um."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def _create_body(self, entry):
        """Erstellt ein einzelnes body- oder schiff-Objekt aus einem JSON-Eintrag."""
        
        # Wenn is_ship=True, verwende die schiff-Klasse
        if entry.get("is_ship", False):
            print(f"LOADER: Creating ship {entry['name']} with position={entry['position']}, velocity={entry.get('velocity', 'NOT FOUND')}")
            return schiff(
                name=entry["name"],
                position=Vec2(entry["position"][0], entry["position"][1]),
                velocity=Vec2(entry["velocity"][0], entry["velocity"][1]) if "velocity" in entry else Vec2(0, 0),
                color=self.hex_to_rgb(entry["color"]) if "color" in entry else (255, 255, 255)
            )
        
        # Ansonsten erstelle einen normalen body
        return body(
            name=entry["name"],
            mass=entry["mass"],
            radius=entry["radius"],
            position=Vec2(entry["position"][0], entry["position"][1]),
            velocity=Vec2(entry["velocity"][0], entry["velocity"][1]),
            fixed=entry.get("fixed", False),
            semi_major_axis=entry.get("semi_major_axis") if "semi_major_axis" in entry else 0.0,
            eccentricity=entry.get("eccentricity") if "eccentricity" in entry else 0.0,
            theta0=entry.get("theta0", 0.0) if "theta0" in entry else 0.0,
            is_moon_of=None,
            is_ship=entry.get("is_ship", False) if "is_ship" in entry else False,
            color=self.hex_to_rgb(entry["color"]) if "color" in entry else (255, 255, 255),
            atmosphere_color=self.hex_to_rgb(entry["atmosphere_color"]) if "atmosphere_color" in entry else (255, 255, 255),
            has_atmosphere=entry.get("has_atmosphere", False),
            atmos_density=entry.get("atmos_density", 0.0),
            light_intensity=entry.get("light_intensity", 0.0),
            style_seed=entry.get("style_seed"),
            style_mode=entry.get("style_mode"),
            style_shape=entry.get("style_shape")
        )
