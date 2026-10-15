#version 330

// Marken der vorhersagelinie (perlen, zeitstriche, pfeilspitzen): ein quad
// je marke, um u_reach px um ihren ort.
in vec2 a_corner;        // x: 0 / 1, y: -1 / +1
in vec4 i_a;             // ort.xy (GL-px, y oben), richtung.xy
in vec4 i_b;             // u, gross (0/1), -, -

uniform vec2 u_viewport;
uniform float u_reach;

flat out vec4 v_a;
flat out vec4 v_b;

void main() {
    vec2 p = i_a.xy + vec2(a_corner.x * 2.0 - 1.0, a_corner.y) * u_reach;
    gl_Position = vec4(p / u_viewport * 2.0 - 1.0, 0.0, 1.0);
    v_a = i_a;
    v_b = i_b;
}
