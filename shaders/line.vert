#version 120

attribute vec2 a_pos;
uniform vec2 u_viewport;

void main() {
    vec2 ndc = vec2(
        (a_pos.x / u_viewport.x) * 2.0 - 1.0,
        1.0 - (a_pos.y / u_viewport.y) * 2.0
    );
    gl_Position = vec4(ndc, 0.0, 1.0);
}
