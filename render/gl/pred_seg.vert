#version 330

// Vorhersagelinie im pixel-stil (`render/prediction_pixel.py`): ein quad je
// segment, um u_reach px ueber strecke und enden hinaus. Koordinaten: GL-
// pixel des puffers (y nach oben).
in vec2 a_corner;        // x: 0 anfang / 1 ende, y: -1 / +1 seite
in vec4 i_a;             // p0.xy, p1.xy
in vec4 i_b;             // u0, u1 (vorausschau 0..1), a0, a1 (bogen in px)

uniform vec2 u_viewport;
uniform float u_reach;

flat out vec4 v_seg;
flat out vec4 v_par;

void main() {
    vec2 p0 = i_a.xy;
    vec2 p1 = i_a.zw;
    vec2 d = p1 - p0;
    float L = length(d);
    vec2 t = L > 1e-4 ? d / L : vec2(1.0, 0.0);
    vec2 n = vec2(-t.y, t.x);
    vec2 p = mix(p0 - t * u_reach, p1 + t * u_reach, a_corner.x) + n * (a_corner.y * u_reach);
    gl_Position = vec4(p / u_viewport * 2.0 - 1.0, 0.0, 1.0);
    v_seg = i_a;
    v_par = i_b;
}
