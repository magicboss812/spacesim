"""Die physik: absoluter (barycentrischer) raum, SI-einheiten, 2D.

    vec.py                Vec2 und G
    world.py              die maschine -- adaptives RKN4, update_planets/_dynamics
    world_kernels.py      der Numba-schnellpfad dazu, bit-identisch zur referenz
    reference_frames.py   Principia-artige plot-rahmen (reine ANSICHT)
    kernels/              die @njit-kerne der vorhersage: integratoren, Kepler,
                          apsiden-suche, propagation

BEZUGSRAHMEN SIND EINE ZEICHEN-TRANSFORMATION, sonst nichts. Der gespeicherte
zustand eines koerpers ist immer absolut -- nie einen rahmen hineinrechnen.
"""
