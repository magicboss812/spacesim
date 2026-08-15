import json
import re
from pathlib import Path
from typing import Any
from vec import Vec2
from bodies import body, schiff

DEFAULT_CONFIG_FILE = "config.json"


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
            ('move_speed', 'move_speed', float),
            ('zoom_factor', 'zoom_factor', float),
        ])
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
            # nur von test.py ausgewertet (tastenbelegung / ein-aus-verhalten)
            ('quality', None, None),
            ('enabled', None, None),
            ('toggle_num_points', None, None),
            ('length_step_factor', None, None),
            ('precision_step_factor', None, None),
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
            ('show_apsis_markers', 'show_apsis_markers', bool),
            ('apsis_marker_radius_px', 'apsis_marker_radius_px', float),
            ('body_icon_radius_px', 'body_icon_radius_px', float),
            ('ship_velocity_vector_length_px', 'ship_velocity_vector_length_px', float),
            ('reference_trajectories_enabled', 'reference_trajectories_enabled', bool),
            ('reference_trajectories_max_points', 'reference_trajectories_max_points', int),
            ('reference_trajectories_sample_step_s', 'reference_trajectories_sample_step_s', float),
            ('reference_traj_min_screen_px', 'reference_traj_min_screen_px', float),
            ('label_texture_cache_max', '_label_texture_cache_max', int),
            # schriftgroessen werden unten in echte font-objekte umgesetzt
            ('hud_font_size_small', None, None),
            ('hud_font_size_medium', None, None),
        ])

        section = self.section('renderer')
        font_small = self.get_int('renderer.hud_font_size_small') if 'hud_font_size_small' in section else None
        font_medium = self.get_int('renderer.hud_font_size_medium') if 'hud_font_size_medium' in section else None
        if font_small is not None or font_medium is not None:
            try:
                import pygame
                pygame.font.init()
                if font_small is not None:
                    renderer.font_small = pygame.font.SysFont(None, font_small)
                if font_medium is not None:
                    renderer.font_medium = pygame.font.SysFont(None, font_medium)
                # gecachte texturen tragen noch die alte schriftgroesse.
                renderer._label_texture_cache = {}
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
        return renderer

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
            light_intensity=entry.get("light_intensity", 0.0)
        )
