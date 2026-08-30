#version 330

// Die POSITIONS-MARKE eines koerpers (siehe body_icon.py).
//
// Ein einziges quad je marke. Welche zelle ein fragment trifft, entscheidet
// der fragment-shader aus `v_local` -- der ICON-lokalen koordinate. Sie haengt
// an `u_center_px`, und das ist eine gleitkomma-position: das zellmuster kann
// deshalb gar nicht ueber die marke wandern, es bewegt sich mit ihr.
//
// Das quad ist absichtlich groesser als die marke (`u_extent`), weil der halo
// darueber hinausreicht.

in vec2 a_corner;            // geteiltes einheits-quad, -1..1 (_ensure_quad_vbo)

uniform vec2 u_center_px;    // bildschirmposition, top-down, GLEITKOMMA
uniform float u_radius_px;   // radius der marke in pixeln (konstante groesse)
uniform float u_extent;      // vielfaches davon, das das quad abdeckt
uniform vec2 u_viewport;

out vec2 v_local;            // icon-koordinaten, y nach OBEN, +-u_extent

void main() {
    vec2 offset = a_corner * u_extent;

    // Der bildschirm zaehlt y nach unten, die marke nach oben -- dieselbe
    // umrechnung wie in body_surface.vert.
    vec2 pos_px = u_center_px + vec2(offset.x, -offset.y) * u_radius_px;

    v_local = offset;
    gl_Position = vec4(
        (pos_px.x / u_viewport.x) * 2.0 - 1.0,
        1.0 - (pos_px.y / u_viewport.y) * 2.0,
        0.0, 1.0
    );
}
