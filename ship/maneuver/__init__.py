"""Manoeverknoten: planen, vorschau zeichnen, automatisch brennen.

    profile.py   das schubprofil -- DIE quelle fuer dauer und zuendzeitpunkt
    plan.py      knoten und plan (hoechstens fuenf), die orbitale basis
    preview.py   die kette coast -> burn -> coast als gezeichnete linie
    executor.py  der autopilot, der einen knoten wirklich fliegt

Die vollstaendigen begruendungen liegen in `.claude/rules/maneuver.md`.
"""

from .executor import ARMED, BURNING, DONE, IDLE, ManeuverExecutor
from .plan import (
    MAX_NODES,
    MIN_EXECUTABLE_DV,
    ManeuverNode,
    ManeuverPlan,
    burn_direction_world,
    orbital_basis,
)
from .preview import ManeuverPreview, body_state_at, state_on_curve
from .profile import BurnProfile

__all__ = [
    'ARMED', 'BURNING', 'DONE', 'IDLE',
    'MAX_NODES', 'MIN_EXECUTABLE_DV', 'BurnProfile', 'ManeuverExecutor',
    'ManeuverNode',
    'ManeuverPlan', 'ManeuverPreview', 'body_state_at',
    'burn_direction_world', 'orbital_basis', 'state_on_curve',
]
