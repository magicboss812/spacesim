#version 330

// Ein partikel: weiche scheibe, ADDITIV in den hitze-puffer (r = hitze,
// g = kaltes kondensat, b = hitze x temperatur). Wo viele uebereinander
// liegen, wird es heiss -- der kern der fahne entsteht aus der dichte, nicht
// aus einer formel. b / r ist im zusammenbau die hitzegewichtete
// temperatur der zelle.

in vec3 v_val;
out vec4 o_heat;

void main() {
    vec2 d = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(d, d);
    if (r2 > 1.0) {
        discard;
    }
    float w = 1.0 - r2;
    o_heat = vec4(v_val.x * w, v_val.y * w, v_val.x * v_val.z * w, 0.0);
}
