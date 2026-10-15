#version 330

// Zellpass des schiffs (`render/ship_pixel.py`): die formen aus
// `ship/art.py::build_cells` liegen im lokalen schiffsraum (+x nase, +y
// oben, SVG-einheiten) und werden hier gedreht, skaliert und in den
// zellpuffer gelegt -- ein fragment je zelle.

in vec2 a_pos;
in vec3 a_col;
in float a_tag;

uniform vec2 u_head;     // nasenrichtung (cos, sin), y nach oben
uniform float u_scale;   // pixel je SVG-einheit
uniform vec2 u_center;   // schiffsmitte relativ zur ecke des puffers, px, y oben
uniform float u_cell;    // zellkante in px
uniform vec2 u_grid;     // puffergroesse in zellen

out vec2 v_local;
out vec3 v_col;
out float v_tag;

void main() {
    vec2 p = vec2(u_head.x * a_pos.x - u_head.y * a_pos.y,
                  u_head.y * a_pos.x + u_head.x * a_pos.y) * u_scale + u_center;
    gl_Position = vec4(p / (u_cell * u_grid) * 2.0 - 1.0, 0.0, 1.0);
    v_local = a_pos;
    v_col = a_col;
    v_tag = a_tag;
}
