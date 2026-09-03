"""Die himmelskoerper: daten, aussehen, bahnlinien.

    body.py          `body` / `schiff` und kepler_relative_xy()
    style.py         der prozedurale vektor-look der koerper
    icon.py          die gesaete pixel-marke, zu der ein koerper herauszoomt
    orbit_lines.py   die bahnlinien (reines numpy, kein GL)

Das ZEICHNEN liegt nicht hier, sondern in `render/bodies.py` -- diese module
erzeugen geometrie und farben, sie reden nicht mit OpenGL.
"""
