"""Kern der UI-schicht: rechtecke, verankerung, widget-basis, eingabe-routing.

ARCHITEKTUR: retained mode mit verankertem layout.
Die HUD-struktur ist statisch, und weiche hover-/panel-uebergaenge sind im
reinen immediate mode muehsam (jedes widget braeuchte einen extern gehaltenen
animationszustand). Widgets halten stattdessen selbst ihren zustand und
bekommen pro frame layout -> update -> draw.

VERANKERUNG ist das, was aufloesungsunabhaengigkeit ueberhaupt erreicht:
ein widget merkt sich eine ECKE/KANTE, einen abstand davon und seine groesse
-- alles in DESIGN-EINHEITEN. Beim resize bleibt es an seiner ecke kleben
statt mitgezogen zu werden.

KONVENTION: die ganze schicht rechnet in TOP-DOWN bildschirmpixeln (ursprung
oben links) -- identisch zu pygames maus-ereignissen. Die umrechnung in die
ortho-konvention passiert in ui/draw.py und ui/text.py, sonst nirgends.
"""

import math

import pygame

from .draw import UIDraw
from .text import TextRenderer
from .theme import DEFAULT_THEME

# Ankerpunkte als (ax, ay) im elternrechteck, 0 = links/oben, 1 = rechts/unten.
TOP_LEFT = (0.0, 0.0)
TOP_CENTER = (0.5, 0.0)
TOP_RIGHT = (1.0, 0.0)
CENTER_LEFT = (0.0, 0.5)
CENTER = (0.5, 0.5)
CENTER_RIGHT = (1.0, 0.5)
BOTTOM_LEFT = (0.0, 1.0)
BOTTOM_CENTER = (0.5, 1.0)
BOTTOM_RIGHT = (1.0, 1.0)

FILL = 'fill'


class Rect:
    """Achsenparalleles rechteck in top-down pixeln."""

    __slots__ = ('x', 'y', 'w', 'h')

    def __init__(self, x=0.0, y=0.0, w=0.0, h=0.0):
        self.x = float(x)
        self.y = float(y)
        self.w = float(w)
        self.h = float(h)

    @property
    def left(self):
        return self.x

    @property
    def top(self):
        return self.y

    @property
    def right(self):
        return self.x + self.w

    @property
    def bottom(self):
        return self.y + self.h

    @property
    def center_x(self):
        return self.x + self.w * 0.5

    @property
    def center_y(self):
        return self.y + self.h * 0.5

    def contains(self, px, py):
        return (
            self.x <= float(px) < self.x + self.w
            and self.y <= float(py) < self.y + self.h
        )

    def inset(self, amount, vertical=None):
        """Nach innen versetztes rechteck (padding)."""
        dx = float(amount)
        dy = float(amount if vertical is None else vertical)
        return Rect(self.x + dx, self.y + dy, max(0.0, self.w - 2 * dx),
                    max(0.0, self.h - 2 * dy))

    def moved(self, dx, dy):
        return Rect(self.x + dx, self.y + dy, self.w, self.h)

    def copy(self):
        return Rect(self.x, self.y, self.w, self.h)

    def __repr__(self):
        return f"Rect({self.x:.1f}, {self.y:.1f}, {self.w:.1f}, {self.h:.1f})"


def ease(current, target, rate, dt):
    """Framerate-unabhaengiges exponentielles easing -- dieselbe formel wie
    das kamera-easing aus Phase 1 (1 - exp(-rate * dt))."""
    if rate <= 0.0:
        return float(target)
    alpha = 1.0 - math.exp(-float(rate) * max(0.0, float(dt)))
    return float(current) + (float(target) - float(current)) * alpha


class UIContext:
    """Alles, was widgets zum zeichnen und messen brauchen.

    Bewusst ein explizites objekt statt globaler zugriffe: dieselbe
    entscheidung wie bei devui.DevContext.
    """

    def __init__(self, ctx, width, height, ui_scale=1.0, theme=DEFAULT_THEME,
                 label_cache_max=256):
        self.gl = ctx
        self.width = int(width)
        self.height = int(height)
        self.ui_scale = float(ui_scale)
        self.theme = theme

        self.draw = UIDraw(ctx, width, height)
        self.text = TextRenderer(
            ctx, width, height, theme=theme, ui_scale=ui_scale,
            cache_max=label_cache_max,
        )
        # Text laeuft ueber eine eigene pipeline. Damit die schichtung mit
        # den GESAMMELTEN rechtecken (UIDraw zeichnet instanziert erst beim
        # flush) exakt der aufruf-reihenfolge entspricht, muss vor jedem
        # text-draw der rechteck-stapel raus.
        self.text.rect_flush = self.draw.flush

        # Eingabezustand des aktuellen frames.
        self.mouse_x = 0.0
        self.mouse_y = 0.0
        self.mouse_down = False
        self.dt = 0.0

    def px(self, design_units):
        """Design-einheiten -> pixel. Das gegenstueck zu Renderer.ui_px().

        Nimmt auch eine FOLGE, denn ein eckradius darf pro ecke verschieden
        sein (theme.cut_corners). Ohne diesen fall muesste jeder aufrufer
        die umrechnung selbst ueber das tupel ziehen -- und genau das ging
        einmal schief: Panel reichte das tupel ungeprueft weiter und starb
        an "float() argument must be ... not 'tuple'".
        """
        if isinstance(design_units, (tuple, list)):
            return tuple(float(value) * self.ui_scale for value in design_units)
        return float(design_units) * self.ui_scale

    @property
    def screen_rect(self):
        return Rect(0.0, 0.0, float(self.width), float(self.height))

    def resize(self, width, height, ui_scale=None):
        self.width = int(width)
        self.height = int(height)
        if ui_scale is not None:
            self.ui_scale = float(ui_scale)
        self.draw.resize(width, height)
        self.text.resize(width, height, ui_scale=self.ui_scale)

    def release(self):
        self.draw.release()
        self.text.release()


class Widget:
    """Basisklasse. Haelt verankerung, groesse, kinder und hover-zustand.

    Groessen und abstaende sind DESIGN-EINHEITEN, nicht pixel -- die
    umrechnung passiert in layout() ueber ctx.px().
    """

    def __init__(self, anchor=TOP_LEFT, offset=(0.0, 0.0), size=(0.0, 0.0),
                 visible=True, enabled=True, z=0, name=None):
        self.anchor = anchor
        self.offset = (float(offset[0]), float(offset[1]))
        self.size = size
        self.visible = bool(visible)
        self.enabled = bool(enabled)
        self.z = int(z)
        self.name = name or self.__class__.__name__

        self.parent = None
        self.children = []
        self.rect = Rect()

        # Ob dieses widget maus-ereignisse VERBRAUCHT. Container sind
        # standardmaessig durchlaessig, panels und bedienelemente nicht --
        # sonst schwenkt ein klick auf ein panel die kamera darunter.
        self.blocks_mouse = False
        self.takes_keyboard = False

        # Interaktionszustaende, von UIRoot gesetzt.
        self.hovered = False
        self.pressed = False
        self.focused = False
        self._hover_t = 0.0
        self._press_t = 0.0

    # ---------------------------------------------------------------- baum

    def add(self, child):
        child.parent = self
        self.children.append(child)
        return child

    def remove(self, child):
        if child in self.children:
            child.parent = None
            self.children.remove(child)

    def walk(self):
        """Sich selbst und alle nachkommen, eltern zuerst."""
        yield self
        for child in self.children:
            for node in child.walk():
                yield node

    # -------------------------------------------------------------- layout

    def measure(self, ctx):
        """Eigengroesse in PIXELN fuer size-komponenten, die None sind.
        Ueberschreiben, wo die groesse aus dem inhalt folgt (labels)."""
        return (0.0, 0.0)

    def _resolve_size(self, ctx, parent_rect):
        """Loest die groessen-angabe auf.

        Pro achse sind drei angaben erlaubt: eine zahl (design-einheiten),
        FILL (elternflaeche fuellen) oder None (eigengroesse aus measure()).
        measure() wird hoechstens einmal aufgerufen -- bei labels ist das
        eine textur-rasterung, die nicht doppelt anfallen soll.
        """
        want_w, want_h = self.size if self.size is not None else (None, None)
        measured = self.measure(ctx) if (want_w is None or want_h is None) else None

        def resolve(value, available, own):
            if value == FILL:
                return float(available)
            if value is None:
                return float(own)
            return float(ctx.px(value))

        return (
            resolve(want_w, parent_rect.w, measured[0] if measured else 0.0),
            resolve(want_h, parent_rect.h, measured[1] if measured else 0.0),
        )

    def desired_size(self, ctx, available=None):
        """Effektive groesse in pixeln: explizite vorgabe, sonst measure().

        Container MUESSEN das hier benutzen und nicht measure() direkt --
        measure() liefert nur die EIGENgroesse und weiss nichts von einem
        gesetzten size=. Ein container, der measure() abfragt, misst ein
        widget mit fester groesse als 0x0 und stapelt alles uebereinander.
        """
        if available is None:
            available = ctx.screen_rect
        return self._resolve_size(ctx, available)

    def layout(self, ctx, parent_rect):
        """Berechnet self.rect aus anker, abstand und groesse."""
        w, h = self._resolve_size(ctx, parent_rect)
        ax, ay = self.anchor

        # Der abstand zeigt IMMER nach innen: an einer rechten kante schiebt
        # ein positiver x-abstand nach links. Ohne diese spiegelung muesste
        # jedes rechts verankerte widget negative zahlen fuehren.
        sign_x = -1.0 if ax > 0.5 else 1.0
        sign_y = -1.0 if ay > 0.5 else 1.0

        x = parent_rect.x + ax * parent_rect.w - ax * w + sign_x * ctx.px(self.offset[0])
        y = parent_rect.y + ay * parent_rect.h - ay * h + sign_y * ctx.px(self.offset[1])

        self.rect = Rect(x, y, w, h)
        self.layout_children(ctx)

    def layout_children(self, ctx):
        content = self.content_rect(ctx)
        for child in self.children:
            child.layout(ctx, content)

    def content_rect(self, ctx):
        """Flaeche, in der kinder verankert werden. Panels ziehen hier ihr
        padding ab."""
        return self.rect

    # -------------------------------------------------------------- update

    def update(self, ctx, dt):
        motion = ctx.theme.motion
        self._hover_t = ease(self._hover_t, 1.0 if self.hovered else 0.0,
                             motion.fast, dt)
        self._press_t = ease(self._press_t, 1.0 if self.pressed else 0.0,
                             motion.fast, dt)
        for child in self.children:
            if child.visible:
                child.update(ctx, dt)

    # ------------------------------------------------------------ zeichnen

    def draw(self, ctx):
        """Eigene darstellung. Basisklasse zeichnet nichts."""

    def draw_tree(self, ctx):
        if not self.visible:
            return
        self.draw(ctx)
        for child in sorted(self.children, key=lambda c: c.z):
            child.draw_tree(ctx)

    # -------------------------------------------------------------- eingabe

    def hit_test(self, ctx, x, y):
        """Trefferflaeche. Bekommt ctx, weil sie nicht immer gleich self.rect
        ist -- ein offenes aufklappmenue nimmt auch treffer in seiner liste
        entgegen, die ausserhalb des eigenen rechtecks liegt."""
        return self.visible and self.rect.contains(x, y)

    def on_mouse_down(self, ctx, x, y, button):
        return False

    def on_mouse_up(self, ctx, x, y, button):
        return False

    def on_mouse_move(self, ctx, x, y):
        return False

    def on_wheel(self, ctx, dx, dy):
        return False

    def on_key(self, ctx, event):
        return False

    def dismiss(self):
        """Woanders wurde geklickt. Aufklappmenues schliessen sich hier."""


class UIRoot(Widget):
    """Wurzel des widget-baums. Verteilt eingaben und meldet, ob sie
    verbraucht wurden.

    EINGABE-VORFAHRT: custom-UI -> ImGui -> welt. Diese klasse ist die
    erste stufe; test.py fragt wants_mouse / wants_keyboard ab und reicht
    sie an devui, kamera und schiffsteuerung weiter.

    WICHTIG: schiffsteuerung und WASD-schwenk lesen die tastatur per POLLING
    (pygame.key.get_pressed()), nicht ueber ereignisse. Ein ereignis hier zu
    verschlucken genuegt ihnen also nicht -- sie muessen wants_keyboard
    zusaetzlich selbst pruefen.
    """

    def __init__(self, ctx):
        super().__init__(anchor=TOP_LEFT, offset=(0.0, 0.0), size=(FILL, FILL),
                         name='UIRoot')
        self.ui = ctx
        self._hover_widget = None
        self._active_widget = None
        self._focus_widget = None
        self._mouse_pos = (0.0, 0.0)

    # ------------------------------------------------------------ zustaende

    @property
    def wants_mouse(self):
        """Solange ein widget gedrueckt wird, gehoert die maus ihm -- auch
        wenn der zeiger dabei aus dem widget herauswandert (slider-drag)."""
        return self._active_widget is not None or self._hover_widget is not None

    @property
    def wants_keyboard(self):
        widget = self._focus_widget
        return widget is not None and widget.takes_keyboard

    @property
    def hovered_widget(self):
        return self._hover_widget

    def set_focus(self, widget):
        if self._focus_widget is widget:
            return
        if self._focus_widget is not None:
            self._focus_widget.focused = False
        self._focus_widget = widget
        if widget is not None:
            widget.focused = True

    # ------------------------------------------------------------- suchbaum

    def paint_order(self):
        """Alle sichtbaren widgets in ZEICHENREIHENFOLGE, hinten zuerst.

        Sortiert nach (effektivem z, tiefensuch-index). Das effektive z eines
        knotens ist das seines elternteils PLUS sein eigenes: ein z am
        container hebt damit seinen ganzen teilbaum an, und ein einzelnes
        widget kann sich trotzdem ueber geschwister-TEILBAEUME legen.

        Warum global und nicht je ebene: vorher wurden nur geschwister nach z
        sortiert, und der zeichen- wie der treffer-pfad liefen einfach in
        tiefensuch-reihenfolge. Ein aufklappmenue mit z=200 verlor damit
        gegen JEDES panel, das weiter hinten im baum haengt -- die
        palettenauswahl wurde von der system-karte verdeckt und fing deren
        klicks ab, obwohl sie sichtbar darueber lag. Ein z, das nur innerhalb
        einer ebene gilt, ist kein z.

        Die tiefensuche haelt eltern vor ihren kindern, das malprinzip
        (erst flaeche, dann inhalt) bleibt also erhalten. Bei gleichem z
        gewinnt das spaeter hinzugefuegte widget.
        """
        items = []
        counter = 0

        def visit(widget, parent_z):
            nonlocal counter
            if not widget.visible:
                return
            z = parent_z + widget.z
            if widget is not self:
                items.append((z, counter, widget))
                counter += 1
            for child in widget.children:
                visit(child, z)

        visit(self, self.z)
        items.sort(key=lambda item: (item[0], item[1]))
        return [item[2] for item in items]

    def _pick(self, x, y):
        """Oberstes widget unter dem zeiger, das die maus beansprucht.

        Genau die umgekehrte zeichenreihenfolge: was zuletzt gemalt wurde,
        liegt oben und faengt den klick zuerst.
        """
        for widget in reversed(self.paint_order()):
            if widget.blocks_mouse and widget.hit_test(self.ui, x, y):
                return widget
        return None

    # -------------------------------------------------------------- ablauf

    def begin_frame(self, dt):
        ctx = self.ui
        ctx.dt = float(dt)
        ctx.mouse_x, ctx.mouse_y = self._mouse_pos
        # Ausfall-sicherung: geht das MOUSEBUTTONUP verloren (fokuswechsel
        # mitten im ziehen), bleibt `pressed` sonst fuer immer stehen -- und
        # ein regler, der pro frame integriert (HorizonSlider), laeuft dann
        # ungebremst weiter. Steht keine maustaste mehr an, den griff loesen.
        if self._active_widget is not None and not any(pygame.mouse.get_pressed()):
            self._active_widget.pressed = False
            self._active_widget = None
        self.layout(ctx, ctx.screen_rect)
        self._refresh_hover()
        self.update(ctx, dt)

    def render(self):
        # Flach in globaler z-reihenfolge zeichnen, NICHT ueber draw_tree:
        # das sortiert nur je ebene und legt ein aufklappmenue unter jedes
        # panel, das weiter hinten im baum steht (siehe paint_order).
        for widget in self.paint_order():
            widget.draw(self.ui)
        # Erst die gesammelten rechtecke raus (instanzierter draw), dann der
        # aufgeschobene text zuletzt: er darf nie unter einer flaeche
        # verschwinden, die spaeter im baum gezeichnet wird.
        self.ui.draw.flush()
        self.ui.text.flush()

    def _refresh_hover(self):
        x, y = self._mouse_pos
        picked = self._pick(x, y)
        if picked is not self._hover_widget:
            if self._hover_widget is not None:
                self._hover_widget.hovered = False
            self._hover_widget = picked
            if picked is not None:
                picked.hovered = True

    # ------------------------------------------------------------- eingabe

    def handle_event(self, event):
        """True = das ereignis gehoert der UI und darf NICHT weitergereicht
        werden."""
        ctx = self.ui

        if event.type == pygame.MOUSEMOTION:
            self._mouse_pos = (float(event.pos[0]), float(event.pos[1]))
            ctx.mouse_x, ctx.mouse_y = self._mouse_pos
            self._refresh_hover()
            if self._active_widget is not None:
                self._active_widget.on_mouse_move(ctx, *self._mouse_pos)
                return True
            if self._hover_widget is not None:
                self._hover_widget.on_mouse_move(ctx, *self._mouse_pos)
                return True
            return False

        if event.type == pygame.MOUSEBUTTONDOWN:
            self._mouse_pos = (float(event.pos[0]), float(event.pos[1]))
            self._refresh_hover()
            target = self._hover_widget
            # Jeder klick schliesst offene aufklappmenues, ausser dem
            # angeklickten selbst -- sonst bliebe eine liste stehen, sobald
            # man daneben klickt.
            for widget in self.walk():
                if widget is not target and widget is not self:
                    widget.dismiss()
            if target is None:
                # Klick ins leere loest den fokus -- sonst frisst ein einmal
                # fokussiertes eingabefeld die tastatur fuer immer.
                self.set_focus(None)
                return False
            ctx.mouse_down = True
            self._active_widget = target
            target.pressed = True
            self.set_focus(target if target.takes_keyboard else None)
            target.on_mouse_down(ctx, self._mouse_pos[0], self._mouse_pos[1], event.button)
            return True

        if event.type == pygame.MOUSEBUTTONUP:
            self._mouse_pos = (float(event.pos[0]), float(event.pos[1]))
            ctx.mouse_down = False
            active = self._active_widget
            if active is None:
                return False
            active.pressed = False
            self._active_widget = None
            active.on_mouse_up(ctx, self._mouse_pos[0], self._mouse_pos[1], event.button)
            self._refresh_hover()
            return True

        if event.type == pygame.MOUSEWHEEL:
            if self._hover_widget is not None:
                return bool(self._hover_widget.on_wheel(ctx, event.x, event.y))
            return False

        if event.type == pygame.KEYDOWN:
            if self._focus_widget is not None and self._focus_widget.takes_keyboard:
                return bool(self._focus_widget.on_key(ctx, event))
            return False

        return False

    def resize(self, width, height, ui_scale=None):
        self.ui.resize(width, height, ui_scale=ui_scale)
