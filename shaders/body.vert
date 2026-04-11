#version 120

attribute vec2 a_corner;
uniform vec2 u_center_px;
uniform float u_outer_radius_px;
uniform vec2 u_viewport;

varying vec2 v_local;

void main() {
    v_local = a_corner;
    vec2 pos_px = u_center_px + a_corner * u_outer_radius_px;
    vec2 ndc = vec2(
        (pos_px.x / u_viewport.x) * 2.0 - 1.0,
        1.0 - (pos_px.y / u_viewport.y) * 2.0
    );
    gl_Position = vec4(ndc, 0.0, 1.0);
}
