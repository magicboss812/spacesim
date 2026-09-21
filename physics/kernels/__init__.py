"""Die @njit-kerne der bahnvorhersage -- reine funktionen, kein `self`.

    integrators.py   beschleunigung, RK4 / RKN4 / adaptives RKN, leapfrog
    kepler.py        skriptierte koerperbahnen (das EINE bahnmodell)
    apsis.py         Ap/Pe-suche und ihre verfeinerung
    propagate.py     die punktreihen: _compute_distance_points_*
    burn.py          das schubprofil eines manoeverknotens

Numba-regeln: keine Python-objekte, kein pygame, kein OpenGL -- nur einfache
arrays und skalare. `cache=True` bleibt ueberall stehen; ein verschieben der
datei entwertet den plattencache, der erste lauf danach ist deshalb langsam.
"""

import numpy as np


#: Spaltenzahl der punkteliste: x, y, t_abs, vx, vy.
#:
#: Die beiden geschwindigkeits-spalten machen aus der liste eine stueckweise
#: KUBISCHE kurve statt einer folge von positionen -- der renderer kann sie
#: damit zur zeichenzeit beliebig fein auswerten (Hermite), ohne dass hier ein
#: integrationsschritt mehr faellt. Die tangente ist die ableitung genau des
#: polynoms, mit dem der rkn-kernel die position interpoliert.
#:
#: Kernel, die ihre punkte LINEAR auf die schrittsehne setzen, schreiben hier
#: NaN. Das ist kein fehlerfall, sondern die wahrheit: ein sehnenpunkt hat
#: keine tangente, die zu ihm passt. Der renderer zeichnet solche abschnitte
#: dann als geraden, statt eine kruemmung zu erfinden.
POINT_COLUMNS = 5

#: Spaltenzahl des koerper-notizblocks: [t, x, y, gueltig] + die fuenf
#: zeitunabhaengigen bahngroessen + deren gueltigkeitsmerker.
#: Siehe `_body_position_at_time_numba`.
BODY_MEMO_COLUMNS = 10


def _no_body_memo():
    """Leerer notizblock fuer aufrufer, bei denen sich das merken nicht lohnt.

    `_body_position_at_time_numba` erkennt an der zeilenzahl 0, dass kein
    notizblock vorliegt, und rechnet ohne.

    **Das MUSS eine funktion sein, keine modul-konstante.** Numba behandelt
    ein globales array als compile-zeit-konstante und typisiert es
    `readonly` -- die (per `use_memo` nie erreichten) schreibzugriffe im
    rumpf lassen sich dann nicht mehr typisieren, und der GANZE aufrufende
    kernel scheitert beim uebersetzen.
    """
    return np.zeros((0, BODY_MEMO_COLUMNS), dtype=np.float64)


def _empty_points():
    """Leere punkteliste in der kanonischen breite."""
    return np.empty((0, POINT_COLUMNS), dtype=np.float64)


def _widen_points(points):
    """Punkte auf POINT_COLUMNS bringen; fehlende tangenten werden NaN."""
    if not isinstance(points, np.ndarray) or points.ndim != 2:
        return points
    have = int(points.shape[1])
    if have >= POINT_COLUMNS:
        return points
    wide = np.empty((points.shape[0], POINT_COLUMNS), dtype=np.float64)
    wide[:, :have] = points
    wide[:, have:] = np.nan
    return wide

from .burn import _burn_arc_numba, _profile_accel_numba  # noqa: E402,F401
