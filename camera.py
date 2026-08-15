import pygame
from vec import Vec2
import math

class Camera:
    """Verwaltet die Ansicht der Simulation: Position, Zoom und Koordinatentransformation.

    Zoom und Schwenk sind *geglättet*: `scale`/`position` sind die tatsächlich
    gezeichneten werte, `target_scale`/`target_position` die ziele, denen sie in
    `update(dt)` exponentiell nachlaufen. Alles, was die kamera von außen
    verstellt (mausrad, ziehen, tasten), schreibt in die ZIELE -- niemals direkt
    in die gezeichneten werte. Ausnahme ist das aktive ziehen, das für direkte
    manipulation 1:1 und ungeglättet sein muss.
    """

    def __init__(self, screen, width, height, sim_dt=9000.0):
        self.screen = screen  # Wird für eventuelle Direktzugriffe behalten
        self.width = width
        self.height = height

        # Simulation time step (seconds per simulation update)
        self.sim_dt = float(sim_dt)
        self.min_sim_dt = 1.0
        self.max_sim_dt = 1e12
        self.sim_dt_factor = 1.5

        # Kamera-Position (Weltkoordinaten des Bildzentrums), gezeichnet + ziel
        self.position = Vec2(0.0, 0.0)
        self.target_position = Vec2(0.0, 0.0)

        # Zoom: Pixel pro Meter (höher = mehr vergrößert), gezeichnet + ziel
        self.scale = 1e-6  # Standard: 1 Pixel = 1.000.000 Meter
        self.target_scale = 1e-6

        # Verfolgtes Objekt (None = freie Kamera)
        self.target = None
        # Welt-versatz zum verfolgten körper. Damit lässt sich beim verfolgen
        # umherziehen, ohne die verfolgung zu verlieren: der körper bleibt
        # angeheftet, die ansicht ist nur verschoben.
        self.follow_offset = Vec2(0.0, 0.0)
        # Gezeichnete (nachlaufende) fassung von follow_offset.
        #
        # WICHTIG: beim verfolgen wird NUR dieser versatz geglättet, niemals die
        # absolute kameraposition. Eine exponentielle glättung auf ein bewegtes
        # ziel behält einen bleibenden rückstand von v/k -- und v ist hier
        # `bahngeschwindigkeit * sim_dt * fps`, also astronomisch: bei sim_dt=900
        # und 17 km/s sind das 9.2e8 m pro echtsekunde, was bei pan_smoothing=20
        # rund 46 px danebenliegt. Der körper stünde dauerhaft neben der
        # bildmitte, und der versatz wüchse linear mit dem zeitraffer.
        self._render_follow_offset = Vec2(0.0, 0.0)

        # Schwenkgeschwindigkeit der tastatur-steuerung, in BILDSCHIRM-HÖHEN
        # pro sekunde. (Früher wurde dieser wert als pixel/sekunde gedeutet --
        # bei einem default von 3.0 schwenkte die ansicht mit 3 px/s, also
        # praktisch gar nicht.)
        self.move_speed = 1.0

        # Zoom-Grenzen
        self.min_scale = 1e-30
        self.max_scale = 1e+10

        # Zoomschritt pro Mausrad-Raste (aus config.json ueberschreibbar)
        self.zoom_factor = 1.5

        # Glättung: höher = schneller/direkter, 0 = aus (sofortiger sprung).
        # Die glättung ist exponentiell und über 1 - exp(-k * dt) formuliert,
        # damit sie bildratenunabhängig ist.
        self.zoom_smoothing = 16.0
        self.pan_smoothing = 20.0

        # Zoom auf den mauszeiger statt auf die bildmitte.
        self.zoom_to_cursor = True

        # Ziehen (mittlere/rechte maustaste) + nachlauf
        self.drag_buttons = (2, 3)
        self._dragging = False
        self._drag_last_screen = None
        self.pan_inertia_enabled = True
        self.pan_inertia_damping = 6.0
        self._pan_velocity = Vec2(0.0, 0.0)   # weltmeter pro sekunde

    # ------------------------------------------------------------------
    # Koordinatentransformation
    # ------------------------------------------------------------------

    def world_to_screen(self, world_pos):
        rel = world_pos - self.position

        # Float-Berechnung
        screen_x = self.width / 2 + rel.x * self.scale
        screen_y = self.height / 2 - rel.y * self.scale

        # PRÜFUNG vor Rückgabe
        if not (math.isfinite(screen_x) and math.isfinite(screen_y)):
            print(f"WARNING: Invalid screen coords: {screen_x}, {screen_y}")
            return (self.width / 2.0, self.height / 2.0)

        # Subpixel-Präzision behalten, damit Bewegung nicht stufig wirkt.
        return (screen_x, screen_y)

    def screen_to_world(self, screen_pos):
        """Wandelt Bildschirmkoordinaten in Weltkoordinaten um."""
        return self._screen_to_world_with(screen_pos, self.position, self.scale)

    def _screen_to_world_with(self, screen_pos, position, scale):
        """screen->welt gegen eine BELIEBIGE position/skala.

        Wird für das zoom-ankern gebraucht: dort muss gegen die ZIEL-werte
        gerechnet werden, nicht gegen die gerade noch nachlaufenden.
        """
        screen_x, screen_y = screen_pos
        safe_scale = max(float(scale), 1e-30)
        world_x = (screen_x - self.width / 2) / safe_scale + position.x
        world_y = -(screen_y - self.height / 2) / safe_scale + position.y
        return Vec2(world_x, world_y)

    # ------------------------------------------------------------------
    # Verfolgung
    # ------------------------------------------------------------------

    def follow(self, target_body):
        """Setzt ein Objekt zur Verfolgung."""
        self.target = target_body
        self.follow_offset.clear()
        self._render_follow_offset.clear()

    def unfollow(self):
        """Beendet die Objektverfolgung.

        Die aktuelle ansicht wird als freies ziel übernommen, damit das lösen
        der verfolgung keinen sprung erzeugt.
        """
        self.target = None
        self.target_position = self.position.copy()
        self.follow_offset.clear()
        self._render_follow_offset.clear()

    def _effective_target_position(self):
        """Weltposition, auf die die kamera gerade zuläuft."""
        if self.target is not None:
            return self.target.position + self.follow_offset
        return self.target_position

    def _shift_target_position(self, delta):
        """Verschiebt das kamera-ziel um `delta` (welt-meter).

        Beim verfolgen wandert der versatz zum körper, sonst das freie ziel --
        so funktioniert dieselbe geste in beiden modi.
        """
        if self.target is not None:
            self.follow_offset += delta
        else:
            self.target_position += delta

    # ------------------------------------------------------------------
    # Zoom
    # ------------------------------------------------------------------

    def zoom_by(self, factor, anchor_screen_pos=None):
        """Multipliziert die ZIEL-skala und verankert optional am mauszeiger.

        Ankern heißt: der weltpunkt unter `anchor_screen_pos` liegt nach dem
        zoom wieder unter derselben bildschirmposition. Gerechnet wird gegen
        die ziel-werte, damit mehrere schnelle rasten sauber aufeinander
        aufbauen statt gegen den nachlaufenden zwischenstand.
        """
        try:
            factor = float(factor)
        except Exception:
            return
        if not (factor > 0.0) or not math.isfinite(factor):
            return

        old_scale = self.target_scale
        new_scale = max(self.min_scale, min(self.max_scale, old_scale * factor))
        if new_scale == old_scale:
            return

        base = self._effective_target_position()
        anchor = self._zoom_anchor(anchor_screen_pos, base, old_scale)

        if anchor is not None:
            anchor_world = self._screen_to_world_with(anchor, base, old_scale)
            # Position bestimmen, bei der anchor_world unter demselben pixel liegt.
            cx, cy = anchor
            safe_new = max(new_scale, 1e-30)
            desired_x = anchor_world.x - (cx - self.width / 2) / safe_new
            desired_y = anchor_world.y + (cy - self.height / 2) / safe_new
            self._shift_target_position(Vec2(desired_x - base.x, desired_y - base.y))

        self.target_scale = new_scale

    def _zoom_anchor(self, cursor_screen_pos, base, scale):
        """Bildschirmpunkt, der beim zoomen stehenbleiben soll (oder None).

        Beim VERFOLGEN ist das immer der verfolgte koerper -- nicht der
        mauszeiger. Ankert man beim verfolgen auf den zeiger, verschiebt jede
        zoom-raste den versatz zum koerper ein stueck weiter, und das schiff
        wandert aus der bildmitte heraus. Auf den koerper geankert bleibt es
        auf demselben pixel und die bildkomposition ueberlebt den zoom.
        """
        if self.target is not None:
            body = self.target.position
            return (
                self.width / 2 + (body.x - base.x) * scale,
                self.height / 2 - (body.y - base.y) * scale,
            )
        if self.zoom_to_cursor:
            return cursor_screen_pos
        return None

    # ------------------------------------------------------------------
    # Ziehen
    # ------------------------------------------------------------------

    def _begin_drag(self, screen_pos):
        self._dragging = True
        self._drag_last_screen = (float(screen_pos[0]), float(screen_pos[1]))
        self._pan_velocity.clear()

    def _update_drag(self, screen_pos, dt):
        if not self._dragging or self._drag_last_screen is None:
            return
        px, py = self._drag_last_screen
        cx, cy = float(screen_pos[0]), float(screen_pos[1])
        self._drag_last_screen = (cx, cy)

        safe_scale = max(self.scale, 1e-30)
        # Bildschirm-delta in welt-delta. Die ansicht folgt dem inhalt unter dem
        # cursor, deshalb gegenläufig zur mausbewegung.
        dx = -(cx - px) / safe_scale
        dy = (cy - py) / safe_scale
        if dx == 0.0 and dy == 0.0:
            return

        delta = Vec2(dx, dy)
        self._shift_target_position(delta)
        # Direkte manipulation: ungeglättet mitziehen, sonst gummibandet es.
        # Beim verfolgen ist position abgeleitet, also muss der GEZEICHNETE
        # versatz mitwandern, nicht die position selbst.
        if self.target is not None:
            self._render_follow_offset += delta
            self.position = self.target.position + self._render_follow_offset
        else:
            self.position += delta

        if dt > 1e-6:
            self._pan_velocity.set(dx / dt, dy / dt)

    def _end_drag(self):
        self._dragging = False
        self._drag_last_screen = None
        if not self.pan_inertia_enabled:
            self._pan_velocity.clear()

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------

    def update(self, dt, ui_wants_keyboard=False):
        """Aktualisiert Zoom und Kameraposition (im Loop aufrufen)."""
        dt = max(0.0, float(dt))

        if not ui_wants_keyboard:
            self._apply_keyboard_pan(dt)

        if self._dragging:
            # Beim ziehen wurde position bereits 1:1 gesetzt; hier nur die
            # skala nachführen.
            self._ease_scale(dt)
            return

        self._apply_pan_inertia(dt)
        self._ease_scale(dt)
        self._ease_position(dt)

    def _apply_keyboard_pan(self, dt):
        """WASD-schwenk. Die pfeiltasten steuern ausschließlich das schiff."""
        keys = pygame.key.get_pressed()
        move_x = float(keys[pygame.K_d]) - float(keys[pygame.K_a])
        move_y = float(keys[pygame.K_w]) - float(keys[pygame.K_s])
        if move_x == 0.0 and move_y == 0.0:
            return

        # move_speed ist in bildschirm-höhen pro sekunde angegeben; über die
        # aktuelle skala in weltmeter umrechnen, damit der schwenk bei jeder
        # zoomstufe gleich schnell WIRKT.
        scale_safe = max(self.scale, 1e-30)
        step = (self.move_speed * float(self.height) / scale_safe) * dt
        self._shift_target_position(Vec2(move_x * step, move_y * step))

    def _apply_pan_inertia(self, dt):
        if not self.pan_inertia_enabled or dt <= 0.0:
            return
        if self._pan_velocity.magnitude_squared() <= 0.0:
            return

        self._shift_target_position(self._pan_velocity * dt)
        damping = math.exp(-max(0.0, self.pan_inertia_damping) * dt)
        self._pan_velocity *= damping
        # Unter einem halben pixel pro sekunde ist der nachlauf unsichtbar.
        if self._pan_velocity.magnitude() * max(self.scale, 1e-30) < 0.5:
            self._pan_velocity.clear()

    def _smoothing_alpha(self, rate, dt):
        """Bildratenunabhängiger glättungsfaktor in [0, 1]."""
        if rate <= 0.0 or dt <= 0.0:
            return 1.0
        return 1.0 - math.exp(-rate * dt)

    def _ease_scale(self, dt):
        target = max(self.min_scale, min(self.max_scale, self.target_scale))
        self.target_scale = target
        if self.scale <= 0.0:
            self.scale = target
            return
        alpha = self._smoothing_alpha(self.zoom_smoothing, dt)
        if alpha >= 1.0:
            self.scale = target
            return
        # Im LOG-raum interpolieren: zoom ist multiplikativ, eine lineare
        # annäherung würde beim rauszoomen kriechen und beim reinzoomen rasen.
        log_now = math.log(self.scale)
        log_target = math.log(target)
        log_now += (log_target - log_now) * alpha
        new_scale = math.exp(log_now)
        # Auf dem letzten promille einrasten, sonst läuft es ewig nach.
        if abs(log_target - log_now) < 1e-4:
            new_scale = target
        self.scale = max(self.min_scale, min(self.max_scale, new_scale))

    def _ease_position(self, dt):
        alpha = self._smoothing_alpha(self.pan_smoothing, dt)

        if self.target is not None:
            # Verfolgung ist EXAKT: der körper sitzt immer genau dort, wo er
            # hingehört. Geglättet wird ausschließlich der vom benutzer
            # erzeugte versatz (ziehen, nachlauf) -- der ist beim loslassen
            # konstant, also läuft er sauber aus, ohne bleibenden rückstand.
            delta = self.follow_offset - self._render_follow_offset
            if alpha >= 1.0 or delta.magnitude() * max(self.scale, 1e-30) < 0.5:
                self._render_follow_offset = self.follow_offset.copy()
            else:
                self._render_follow_offset += delta * alpha
            self.position = self.target.position + self._render_follow_offset
            return

        desired = self.target_position
        if alpha >= 1.0:
            self.position = desired.copy()
            return
        delta = desired - self.position
        self.position += delta * alpha
        # Unter einem halben pixel abstand einrasten.
        if delta.magnitude() * max(self.scale, 1e-30) < 0.5:
            self.position = desired.copy()

    # ------------------------------------------------------------------
    # Eingabe
    # ------------------------------------------------------------------

    def handle_event(self, event, ui_wants_mouse=False, ui_wants_keyboard=False):
        """Verarbeitet Eingabeereignisse (Zoom, Ziehen, Klicks).

        `ui_wants_mouse` / `ui_wants_keyboard` sind die eingabe-vorfahrt: liegt
        der zeiger über einer oberfläche (custom-UI, ImGui), darf die kamera
        das ereignis nicht mehr verbrauchen.
        """
        if event.type == pygame.MOUSEWHEEL:
            if ui_wants_mouse:
                return
            # Mausrad für Zoom. `precise_y` (touchpads, feine räder) bevorzugen
            # und die stärke der raste auswerten, damit ein schneller stoß
            # weiter zoomt als eine einzelne raste.
            step = max(float(getattr(self, "zoom_factor", 1.5)), 1.0 + 1e-9)
            notches = float(getattr(event, 'precise_y', 0.0) or 0.0)
            if notches == 0.0:
                notches = float(getattr(event, 'y', 0) or 0)
            if notches == 0.0:
                return
            self.zoom_by(step ** notches, anchor_screen_pos=pygame.mouse.get_pos())

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if ui_wants_mouse:
                return
            if event.button in self.drag_buttons:
                self._begin_drag(event.pos)

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button in self.drag_buttons and self._dragging:
                self._end_drag()

        elif event.type == pygame.MOUSEMOTION:
            if self._dragging:
                # Ziehen läuft weiter, auch wenn der zeiger über die UI wandert:
                # eine begonnene geste gehört dem, der sie begonnen hat.
                self._update_drag(event.pos, self._last_motion_dt())

        elif event.type == pygame.KEYDOWN:
            if ui_wants_keyboard:
                return
            # Verfolgung umschalten
            if event.key == pygame.K_f:
                if self.target is not None:
                    self.unfollow()
            # Ansicht auf den verfolgten körper zurückzentrieren
            elif event.key == pygame.K_HOME:
                self.follow_offset.clear()
                self._pan_velocity.clear()
            # simulation timestep steuerung (PageUp/PageDown)
            elif event.key == pygame.K_PAGEUP:
                self.sim_dt *= self.sim_dt_factor
                self.sim_dt = min(self.sim_dt, self.max_sim_dt)
            elif event.key == pygame.K_PAGEDOWN:
                self.sim_dt /= self.sim_dt_factor
                self.sim_dt = max(self.sim_dt, self.min_sim_dt)

    def _last_motion_dt(self):
        """Zeit seit dem letzten MOUSEMOTION, für die nachlauf-geschwindigkeit."""
        now = pygame.time.get_ticks() / 1000.0
        previous = getattr(self, '_last_motion_time', None)
        self._last_motion_time = now
        if previous is None:
            return 0.0
        return max(0.0, now - previous)

    def snap_to_targets(self):
        """Übernimmt zoom/position sofort (kein nachlauf). Für initialisierung."""
        self.scale = max(self.min_scale, min(self.max_scale, self.target_scale))
        self._render_follow_offset = self.follow_offset.copy()
        self.position = self._effective_target_position().copy()
        self._pan_velocity.clear()
