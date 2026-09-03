import pygame
from physics.vec import Vec2
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

        # Verfolgtes Objekt (None = freie Kamera).
        #
        # Die verfolgung ist ENTWEDER-ODER: solange sie besteht, sitzt der
        # körper EXAKT in der bildmitte, und jeder schwenk (WASD, ziehen) löst
        # sie auf. Es gibt keinen zustand "verfolgt, aber verschoben" mehr --
        # früher gab es ihn (`follow_offset`), und er machte den zeitraffer
        # unbrauchbar: ein versatz zum körper wird ja mitgeführt, die kamera
        # rennt also mit bahngeschwindigkeit durchs system, obwohl der spieler
        # nur zur seite geschaut hat. Nach dem schwenk steht die kamera stattdessen
        # im weltraum still (weltgeschwindigkeit exakt 0); angeheftet wird nur
        # wieder, wenn der spieler einen körper anwählt oder Home drückt.
        self.target = None
        # Restversatz eines LAUFENDEN anflugs (`focus_on`), der auf null
        # ausläuft. Nur dieser versatz wird geglättet, niemals die absolute
        # kameraposition: eine exponentielle glättung auf ein BEWEGTES ziel
        # behält einen bleibenden rückstand von v/k -- und v ist hier
        # `bahngeschwindigkeit * sim_dt * fps`, also astronomisch: bei sim_dt=900
        # und 17 km/s sind das 9.2e8 m pro echtsekunde, was bei pan_smoothing=20
        # rund 46 px danebenliegt. Der körper stünde dauerhaft neben der
        # bildmitte, und der versatz wüchse linear mit dem zeitraffer.
        self._focus_offset = Vec2(0.0, 0.0)

        # Schwenkgeschwindigkeit der tastatur-steuerung, in BILDSCHIRM-HÖHEN
        # pro sekunde. (Früher wurde dieser wert als pixel/sekunde gedeutet --
        # bei einem default von 3.0 schwenkte die ansicht mit 3 px/s, also
        # praktisch gar nicht.)
        self.move_speed = 1.0

        # Zoom-Grenzen
        self.min_scale = 1e-30
        self.max_scale = 1e+10
        #: Zusaetzliche, PHYSIKALISCHE obergrenze des hineinzoomens: der
        #: schirm zeigt nie weniger als so viele meter in der breite. `scale`
        #: ist px je meter und haengt an der fensterbreite, eine feste
        #: `max_scale` waere also bei jeder aufloesung eine andere strecke.
        #: 0 oder None schaltet die grenze ab.
        self.min_visible_span_m = 12.0

        # Zoomschritt pro Mausrad-Raste (aus config.json ueberschreibbar)
        self.zoom_factor = 1.5

        # Glättung: höher = schneller/direkter, 0 = aus (sofortiger sprung).
        # Die glättung ist exponentiell und über 1 - exp(-k * dt) formuliert,
        # damit sie bildratenunabhängig ist.
        self.zoom_smoothing = 16.0
        self.pan_smoothing = 20.0
        # Eigene, LANGSAMERE rate für den anflug auf einen angeklickten körper
        # (`focus_on`). pan_smoothing = 20 ist in ~0.15 s fertig -- das liest
        # sich als sprung, nicht als fahrt. 4.5 sind rund 0.7 s.
        self.focus_smoothing = 4.5
        self._focus_active = False

        # Körper, zu dem `Home` zurückkehrt (normalerweise das schiff).
        self.home_body = None

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
        """Setzt ein Objekt zur Verfolgung (SOFORT, ohne anflug)."""
        self.target = target_body
        self._focus_offset.clear()
        self._focus_active = False

    def set_home_body(self, body):
        """Körper, auf den `Home` die ansicht zurückholt (das schiff).

        Die kamera kennt die weltinhalte sonst nicht; ohne diesen einen
        expliziten verweis müsste die haupt schleife die taste abfangen,
        bevor `handle_event` sie sieht -- zwei stellen für eine taste.
        """
        self.home_body = body

    def focus_on(self, body):
        """Fährt die ansicht GEGLÄTTET auf `body` und heftet sie dort an.

        Der trick ist, dass hier NICHTS neu geglättet wird: die kamera
        springt sofort auf den neuen körper um, und `_focus_offset` wird auf
        genau die differenz gesetzt, die das bild unverändert lässt.
        `_ease_position` läuft ihn dann auf null zu.

        Warum nicht direkt die absolute position auf den körper zu glätten:
        das ziel bewegt sich (bahngeschwindigkeit x zeitraffer), und eine
        exponentielle glättung auf ein bewegtes ziel behält einen bleibenden
        rückstand von v/k -- siehe die anmerkung an `_focus_offset`.
        Der versatz dagegen läuft auf die KONSTANTE null zu, also endet der
        anflug exakt auf dem körper, bei jeder raffungsstufe.

        Bezugsrahmen: der bildmittelpunkt ist `frame(camera.position)`, und
        körper wie kamera gehen durch dieselbe transformation. Ein versatz in
        WELTkoordinaten ist deshalb im nicht-rotierenden rahmen 1:1 ein
        bildschirmversatz, im richtungsrahmen zusätzlich mit der bahnrate des
        bezugskörpers gedreht -- über die ~0.7 s des anflugs sind das bei der
        Erde ~1e-7 rad. Eine rücktransformation ist hier also nicht nötig.
        """
        if body is None:
            return
        previous = self.position.copy()
        self.target = body
        self._focus_offset = previous - body.position
        self._pan_velocity.clear()
        self._focus_active = True

    def recentre(self):
        """Taste Home: geglättet zurück zum heimatkörper (dem schiff)."""
        self.focus_on(self.home_body)

    def unfollow(self):
        """Beendet die Objektverfolgung.

        Die aktuelle ansicht wird als freies ziel übernommen, damit das lösen
        der verfolgung keinen sprung erzeugt -- und weil `target_position`
        danach konstant ist, steht die kamera im weltraum still, statt die
        bahngeschwindigkeit des körpers zu erben.
        """
        self.target = None
        self.target_position = self.position.copy()
        self._focus_offset.clear()
        self._focus_active = False

    def _scale_ceiling(self):
        """Die tatsaechliche zoom-obergrenze (px je meter).

        `max_scale` ist die harte, konfigurierte decke; `min_visible_span_m`
        legt zusaetzlich fest, wie wenig welt der schirm hoechstens zeigen
        darf. Beides zusammen, damit ein groesseres fenster nicht automatisch
        weiter hineinzoomen laesst.
        """
        ceiling = float(self.max_scale)
        span = getattr(self, 'min_visible_span_m', 0.0) or 0.0
        if span > 0.0 and self.width > 0:
            ceiling = min(ceiling, float(self.width) / float(span))
        return max(ceiling, float(self.min_scale))

    def _effective_target_position(self):
        """Weltposition, auf die die kamera gerade zuläuft."""
        if self.target is not None:
            return self.target.position
        return self.target_position

    def _shift_target_position(self, delta):
        """Schwenkt die ansicht um `delta` (welt-meter) -- und LÖST DABEI.

        Ein schwenk ist die aussage "ich will woanders hinsehen", nicht "ich
        will den körper weiter verfolgen, nur versetzt". Der zweite zustand
        existierte einmal (`follow_offset`) und war im zeitraffer unbrauchbar:
        er führt den versatz mit, die kamera fliegt also mit voller
        bahngeschwindigkeit weiter, obwohl der spieler nur zur seite geschaut
        hat -- und beim zoomen sieht man den versatz nachlaufen, als hinge die
        kamera dem schiff hinterher.

        Das lösen passiert genau EINMAL, beim ersten schwenk: danach ist
        `target` schon None. Ohne diese bedingung würde `unfollow()` in jedem
        frame `target_position` auf die (noch nachlaufende) position
        zurücksetzen und die glättung damit aushebeln.
        """
        if self.target is not None:
            self.unfollow()
        self.target_position += delta

    # ------------------------------------------------------------------
    # Zoom
    # ------------------------------------------------------------------

    def zoom_by(self, factor):
        """Multipliziert die ZIEL-skala. Der zoom ankert an der BILDMITTE.

        Das heißt: er verschiebt die kamera überhaupt nicht, er ändert nur die
        skala. Damit ist der zoom die einzige geste, die den bildmittelpunkt
        garantiert in ruhe lässt -- was beim verfolgen genau das gewünschte
        ergebnis hat: der körper steht still, das bild wächst um ihn herum.

        Vorher wurde auf den MAUSZEIGER geankert (die karten-konvention). Das
        verschiebt das kamera-ziel bei jeder raste, und bei schnellem
        auf-und-ab-zoomen sieht man die geglättete position dem ziel
        hinterherlaufen -- als würde die kamera versuchen, das schiff
        einzuholen. Genau dieses nachlaufen ist der grund, warum es weg ist;
        wer eine andere stelle betrachten will, schwenkt dorthin.

        Gerechnet wird gegen `target_scale`, nicht gegen die nachlaufende
        `scale`: so bauen mehrere schnelle rasten sauber aufeinander auf.
        """
        try:
            factor = float(factor)
        except Exception:
            return
        if not (factor > 0.0) or not math.isfinite(factor):
            return

        self.target_scale = max(self.min_scale,
                                min(self._scale_ceiling(),
                                    self.target_scale * factor))

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
        # Löst die verfolgung beim ersten schritt (siehe _shift_target_position)
        # und verschiebt danach nur noch das freie ziel.
        self._shift_target_position(delta)
        # Direkte manipulation: ungeglättet mitziehen, sonst gummibandet es.
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
        target = max(self.min_scale,
                     min(self._scale_ceiling(), self.target_scale))
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
        self.scale = max(self.min_scale,
                         min(self._scale_ceiling(), new_scale))

    def _ease_position(self, dt):
        # Waehrend eines anflugs (`focus_on` / `recentre`) laeuft der versatz
        # mit der langsameren focus-rate aus; danach wieder mit pan_smoothing,
        # damit ziehen und nachlauf direkt bleiben.
        rate = self.focus_smoothing if self._focus_active else self.pan_smoothing
        alpha = self._smoothing_alpha(rate, dt)

        if self.target is not None:
            # Verfolgung ist EXAKT: der körper sitzt immer genau in der
            # bildmitte. Geglättet wird ausschließlich der restversatz eines
            # laufenden anflugs, und dessen ziel ist die KONSTANTE null --
            # deshalb bleibt kein rückstand, egal wie schnell der körper fliegt.
            #
            # Ausserhalb eines anflugs ist `_focus_offset` exakt (0, 0) und
            # `position` damit bitgenau `target.position`. Genau das macht den
            # zoom bewegungsfrei: es gibt nichts, was nachlaufen könnte.
            if alpha >= 1.0 or (self._focus_offset.magnitude()
                                * max(self.scale, 1e-30) < 0.5):
                self._focus_offset.clear()
                # Angekommen: der anflug ist vorbei, ab hier wieder direkt.
                self._focus_active = False
            else:
                self._focus_offset -= self._focus_offset * alpha
            self.position = self.target.position + self._focus_offset
            return

        desired = self.target_position
        if alpha >= 1.0:
            self.position = desired.copy()
            self._focus_active = False
            return
        delta = desired - self.position
        self.position += delta * alpha
        # Unter einem halben pixel abstand einrasten.
        if delta.magnitude() * max(self.scale, 1e-30) < 0.5:
            self.position = desired.copy()
            self._focus_active = False

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
            self.zoom_by(step ** notches)

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
            # Home holt die ansicht zum heimatkoerper (schiff) zurueck -- der
            # weg zurueck, nachdem ein schwenk die verfolgung geloest hat.
            elif event.key == pygame.K_HOME:
                self.recentre()
            # simulation timestep steuerung (PageUp/PageDown)
            elif event.key == pygame.K_PAGEUP:
                self.sim_dt *= self.sim_dt_factor
                self.sim_dt = min(self.sim_dt, self.max_sim_dt)
            elif event.key == pygame.K_PAGEDOWN:
                self.sim_dt /= self.sim_dt_factor
                self.sim_dt = max(self.sim_dt, self.min_sim_dt)

    def allow_warp_rate(self, rate_s_per_s, tick_rate):
        """Senkt den sim_dt-boden, damit diese RAFFUNG erreichbar bleibt.

        `min_sim_dt` ist in sim-sekunden JE TICK angegeben und haengt damit an
        der bildrate; zeitraffer-stufen sind in sim-sekunden je ECHTSEKUNDE
        angegeben und tun das nicht. Der config-wert 1.0 stammt aus der zeit
        von 60 fps, wo er genau die unterste stufe (60 s/s) traf.

        Bei window.fps = 180 sperrte er sie aus: die langsamste erreichbare
        rate war 1.0 * 180 = 180 s/s, also dauerhaft ueber
        simulation.realtime_warp_max. Folge im spiel -- der schub war in JEDER
        stufe gesperrt (der regler zeigte staendig "HOLD") und die vorhersage
        kam nie aus dem zeitraffer-halt heraus. Der boden darf die
        echtzeit-stufe bei keiner bildrate ausschliessen.
        """
        rate = float(rate_s_per_s)
        ticks = float(tick_rate)
        if not (rate > 0.0) or not (ticks > 0.0):
            return
        self.min_sim_dt = min(float(self.min_sim_dt), rate / ticks)
        self.sim_dt = max(float(self.sim_dt), self.min_sim_dt)

    def warp_rate(self, tick_rate):
        """Aktuelle raffung in sim-sekunden je ECHTsekunde.

        `sim_dt` ist je TICK angegeben, die zeitraffer-stufen des HUDs je
        echtsekunde -- diese multiplikation ist der einzige uebergang zwischen
        beiden, und sie stand vorher als closure in `test.py`.
        """
        return float(self.sim_dt) * float(tick_rate)

    def thrust_allowed(self, tick_rate, realtime_warp_max):
        """Schub nur in echtzeit -- siehe .claude/rules/camera-input.md.

        Kleine toleranz, damit die unterste stufe nicht an rundung scheitert.
        """
        return self.warp_rate(tick_rate) <= float(realtime_warp_max) * 1.001

    def clamp_warp_to_timescale(self, t_char, tick_rate, divisor,
                                realtime_warp_max):
        """Raffung auf das begrenzen, was die BAHN noch aufloest.

        Nahe an einem koerper ist die obergrenze keine frage der rechen-
        leistung: ein frame bei 1 y/s rueckt um 48 stunden vor, das sind rund
        24 umlaeufe eines 2-stunden-orbits. Gemessen in einem 2000-km-orbit
        bei 1 y/s: 5120 teilschritte und 270 ms je frame -- und die waeren
        auch dann noetig, wenn sie billig waeren, weil sonst schlicht die
        bahn verloren geht.

        Das HUD blendet gesperrte stufen bereits ab; das hier ist der riegel
        fuer PageUp/PageDown und die dev-oberflaeche, die daran vorbeigehen.
        `t_char` kommt aus `world.characteristic_timescale(ship)`; ist sie
        unbekannt (None oder <= 0), passiert nichts.
        """
        if not t_char or t_char <= 0.0:
            return
        ticks = float(tick_rate)
        cap_rate = max(float(t_char) / float(divisor) * ticks,
                       float(realtime_warp_max))
        if self.warp_rate(ticks) > cap_rate:
            self.sim_dt = max(float(getattr(self, 'min_sim_dt', 1e-6)),
                              cap_rate / ticks)

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
        self.scale = max(self.min_scale,
                         min(self._scale_ceiling(), self.target_scale))
        self._focus_offset.clear()
        self.position = self._effective_target_position().copy()
        self._pan_velocity.clear()
        self._focus_active = False
