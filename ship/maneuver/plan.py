"""Manoeverknoten und der plan, der sie haelt.

EIN KNOTEN SPEICHERT ZWEI ZAHLEN UND EINE ZEIT -- nicht mehr. Brenndauer,
zuendzeitpunkt und die weltrichtung des schubs sind ABGELEITET und werden
bei jedem lesen neu gerechnet (`profile()`, `burn_direction_world()`).
Gespeichert waeren sie genau so lange richtig, bis der spieler das delta-v
verstellt -- und der fehler faellt dann erst beim brennen auf.

DAS VORZEICHEN VON NORMAL IST FESTGELEGT UND WIRD NICHT NEU ERFUNDEN:
positiv heisst NACH INNEN, zum bezugskoerper. Dieselbe regel benutzt
`physics/reference_frames.py::apparent_orbital_directions` fuer
`normal_in`, und daran haengen die snap-rosette, die marker am kompassring
und die ziehgriffe an der linie. Waeren es zwei regeln, zeigte der griff
in die eine und das schiff bruennte in die andere richtung.

Die reihenfolge im plan ist die ZEITLICHE. `first()` ist der naechste
knoten, und genau den nimmt EXECUTE -- 'der erste' heisst nie 'der zuerst
gesetzte'.
"""

import math

from .profile import BurnProfile

#: Ab diesem betrag gilt ein knoten als ausfuehrbar. Darunter ist er ein
#: reiner PLATZHALTER -- man setzt ihn, um eine stelle der bahn zu
#: markieren, und entscheidet erst danach, was dort passieren soll. Der
#: EXECUTE-knopf ist so lange abgeblendet (siehe ui/hud/maneuver.py).
MIN_EXECUTABLE_DV = 1e-3

#: Voreinstellung fuer die plangroesse. Fuenf, weil ein transfer aus
#: abflug, korrektur und einfang besteht und zwei reserve genug sind; jeder
#: weitere knoten haengt an einer bahn, die schon eine integration mehr
#: tief ist (siehe .claude/rules/maneuver.md).
MAX_NODES = 5


def orbital_basis(rel_px, rel_py, rel_vx, rel_vy):
    """Prograde und einwaerts-normal am ort des knotens, in WELTkoordinaten.

    `rel_*` sind position und geschwindigkeit RELATIV zum bezugskoerper.
    Zurueck kommt `(px, py, nx, ny)` -- zwei einheitsvektoren, orthogonal,
    `n` mit positivem anteil in richtung des bezugskoerpers. `None`, wenn
    die relativgeschwindigkeit verschwindet (dann gibt es kein prograde).

    Auf einer rein RADIALEN bahn steht `v` parallel zu `r`; 'einwaerts'
    laesst sich dann nicht mehr aus dem skalarprodukt entscheiden. Die
    funktion liefert trotzdem eine orthonormale basis (die erste
    kandidaten-senkrechte), statt None -- sonst liesse sich auf einer
    radialbahn ueberhaupt kein knoten setzen.
    """
    speed = math.hypot(float(rel_vx), float(rel_vy))
    if speed <= 1e-12:
        return None
    px = float(rel_vx) / speed
    py = float(rel_vy) / speed

    # Die beiden senkrechten. Welche 'einwaerts' ist, entscheidet das
    # skalarprodukt mit der richtung zum bezugskoerper (= -rel_pos).
    nx, ny = -py, px
    r = math.hypot(float(rel_px), float(rel_py))
    if r > 1e-12:
        ix = -float(rel_px) / r
        iy = -float(rel_py) / r
        if nx * ix + ny * iy < 0.0:
            nx, ny = py, -px
    return (px, py, nx, ny)


def burn_direction_world(basis, dv_prograde, dv_normal):
    """Weltrichtung und betrag des geplanten schubs.

    Zurueck kommt `(dx, dy, magnitude)` mit `(dx, dy)` als einheitsvektor.
    `(0.0, 0.0, 0.0)`, wenn keine basis vorliegt oder beide anteile null
    sind.
    """
    if basis is None:
        return (0.0, 0.0, 0.0)
    px, py, nx, ny = basis
    p = float(dv_prograde)
    n = float(dv_normal)
    dx = px * p + nx * n
    dy = py * p + ny * n
    magnitude = math.hypot(dx, dy)
    if magnitude <= 1e-12:
        return (0.0, 0.0, 0.0)
    return (dx / magnitude, dy / magnitude, magnitude)


class ManeuverNode:
    """Ein geplanter impuls: wann, und wieviel prograde/normal."""

    __slots__ = ('t_node', 'dv_prograde', 'dv_normal', 'reference_index')

    def __init__(self, t_node, dv_prograde=0.0, dv_normal=0.0,
                 reference_index=None):
        #: Absolute sim-zeit des knotens -- die MITTE des brennvorgangs,
        #: nicht sein anfang (siehe BurnProfile.lead_time).
        self.t_node = float(t_node)
        self.dv_prograde = float(dv_prograde)
        self.dv_normal = float(dv_normal)
        #: Gegen welchen koerper prograde/normal gemeint sind. None heisst
        #: 'der gerade gewaehlte bezugskoerper'.
        self.reference_index = reference_index

    @property
    def dv_total(self):
        return math.hypot(self.dv_prograde, self.dv_normal)

    def is_executable(self, min_dv=MIN_EXECUTABLE_DV):
        return self.dv_total > float(min_dv)

    def profile(self, a_max, ramp_seconds):
        """Das schubprofil dieses knotens -- IMMER frisch gerechnet."""
        return BurnProfile(self.dv_total, a_max, ramp_seconds)

    def __repr__(self):
        return (f"ManeuverNode(t={self.t_node:.1f}, "
                f"pro={self.dv_prograde:+.1f}, nrm={self.dv_normal:+.1f})")


class ManeuverPlan:
    """Bis zu `max_nodes` knoten, nach zeit sortiert."""

    def __init__(self, max_nodes=MAX_NODES, min_executable_dv=MIN_EXECUTABLE_DV):
        self.max_nodes = max(1, int(max_nodes))
        self.min_executable_dv = float(min_executable_dv)
        self.nodes = []
        #: Zaehlt JEDE aenderung. Vorschau und renderer rechnen nur neu,
        #: wenn diese zahl sich bewegt hat -- eine kette aus fuenf
        #: integrationen je frame waere sonst der teuerste posten im bild.
        self.version = 0

    # ------------------------------------------------------------- abfragen

    def __len__(self):
        return len(self.nodes)

    def __iter__(self):
        return iter(self.nodes)

    @property
    def is_full(self):
        return len(self.nodes) >= self.max_nodes

    def first(self):
        """Der zeitlich naechste knoten -- den fliegt EXECUTE."""
        return self.nodes[0] if self.nodes else None

    def index_of(self, node):
        for i, candidate in enumerate(self.nodes):
            if candidate is node:
                return i
        return None

    # ------------------------------------------------------------ aendern

    def _sort(self):
        self.nodes.sort(key=lambda node: node.t_node)

    def touch(self):
        """Nach einer direkten feldaenderung aufrufen.

        Sortiert nach (ein ziehgriff darf `t_node` verstellen) und zaehlt
        die version hoch. Ohne den aufruf bleibt die gezeichnete linie auf
        dem alten stand stehen.
        """
        self._sort()
        self.version += 1

    def add(self, node):
        if len(self.nodes) >= self.max_nodes:
            return False
        self.nodes.append(node)
        self.touch()
        return True

    def remove(self, node):
        index = self.index_of(node)
        if index is None:
            return False
        return self.remove_at(index)

    def remove_at(self, index):
        if not (0 <= int(index) < len(self.nodes)):
            return False
        self.nodes.pop(int(index))
        self.touch()
        return True

    def clear(self):
        if not self.nodes:
            return False
        self.nodes = []
        self.touch()
        return True
