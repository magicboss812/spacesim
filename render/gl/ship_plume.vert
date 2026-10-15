#version 330

// Abgas-partikel (`ship/plume.py`) als punkte in den hitze-puffer im
// zellraster des schiffs. a_pos: schiffsmitte = 0, achsen wie der schirm
// (y oben), einheiten der bauform; a_attr = (groesse, hitze, kalt,
// temperatur).

in vec2 a_pos;
in vec4 a_attr;

uniform float u_scale;   // pixel je einheit
uniform vec2 u_center;   // schiffsmitte relativ zur pufferecke, px
uniform float u_cell;    // zellkante, px
uniform vec2 u_grid;     // puffer in zellen

out vec3 v_val;

void main() {
    vec2 p = a_pos * u_scale + u_center;
    gl_Position = vec4(p / (u_cell * u_grid) * 2.0 - 1.0, 0.0, 1.0);
    gl_PointSize = max(1.0, 2.0 * a_attr.x * u_scale / u_cell);
    v_val = a_attr.yzw;
}
