"""Das schiff des spielers und was an ihm haengt.

    control.py    schiffcontrol -- tastatur -> drehung, schub, rast-autopilot
    camera.py     welt<->bildschirm, zoom, verfolgung, sim_dt (zeitraffer)
    art.py        die vektor-zeichnung des schiffs
    horizon.py    die laengenregel des vorhersage-horizonts
    predictor.py  die vorausberechnete bahnlinie

Die kamera liegt hier, weil sie standardmaessig am schiff haengt und `Home`
zu ihm zurueckfuehrt -- sie ist teil der spielersicht, nicht der welt.
"""
