"""Die vorausberechnete bahnlinie des schiffs.

    core.py      zustand, integrator-guete, der frame-einstieg update()
    hold.py      der warp-halt und das verbrauchen der kurve
    compute.py   schnappschuss -> integration -> punktreihe
    jobs.py      die asynchrone rechen-pipeline
    view.py      was herauskommt: punkte, Ap/Pe-marker, laenge, abstand

Die reine zahlenarbeit liegt NICHT hier, sondern in `physics/kernels/`.

Die kern-namen werden mit weitergereicht: `tests/warp_predictor_test.py` holt
`_find_apsis_markers_numba` ueber diesen pfad, und der predictor war jahrelang
die adresse dafuer.
"""
from ship.predictor.core import Predictor
from physics.kernels import (
    BODY_MEMO_COLUMNS,
    POINT_COLUMNS,
    _empty_points,
    _no_body_memo,
    _widen_points,
)
from physics.kernels.apsis import _find_apsis_markers_numba, _refine_apsis_numba

__all__ = ['Predictor', 'POINT_COLUMNS', 'BODY_MEMO_COLUMNS',
           '_find_apsis_markers_numba', '_refine_apsis_numba',
           '_empty_points', '_widen_points', '_no_body_memo']
