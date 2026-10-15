#version 330

// Ein segment der vorhersagelinie in den puffer (blending MAX). Gemessen
// wird der abstand der PIXELMITTE zur strecke: so entsteht ein hartes
// pixelband ohne kantenglaettung, und an den fugen nimmt MAX das staerkere
// der beiden segmente statt sie zu stapeln.
//
// Kanaele: r = linie, g = dunkler saum, b = schiene, a = marke.
// Das dithern zaehlt in BAHN-koordinaten (bogen ab einem anker auf einem
// vielfachen der absoluten zeit, quer zur strecke): das muster steht auf der
// bahn still, auch wenn die kamera die linie ueber den schirm schiebt.

flat in vec4 v_seg;
flat in vec4 v_par;

uniform int u_style;       // 1 kontur, 2 perlen, 3 lineal, 4 halbton, 5 schiene
uniform float u_w;         // breite des bands, px
uniform float u_outline;   // saum, px
uniform vec2 u_aux;        // je stil: deckkraft / mindestdeckung / stufen
uniform vec2 u_fade;       // stufen, kleinste deckkraft am ende

out vec4 o_buf;

float bayer4(ivec2 p) {
    int i = (p.x & 3) + 4 * (p.y & 3);
    int m[16] = int[16](0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5);
    return (float(m[i]) + 0.5) / 16.0;
}

// Deckkraft ueber die vorausschau: u_fade.x harte stufen von 1 bis
// u_fade.y, die grenzen gedithert.
float fade(float u, float b) {
    float n = u_fade.x;
    if (n <= 1.0) return 1.0;
    float q = clamp(floor(u * n + (b - 0.5) * 0.6), 0.0, n - 1.0);
    return 1.0 - (1.0 - u_fade.y) * q / (n - 1.0);
}

void main() {
    vec2 c = floor(gl_FragCoord.xy) + 0.5;
    vec2 p0 = v_seg.xy;
    vec2 d = v_seg.zw - p0;
    float L2 = dot(d, d);
    float s = L2 > 1e-8 ? clamp(dot(c - p0, d) / L2, 0.0, 1.0) : 0.0;
    float dist = length(c - (p0 + d * s));
    float side = L2 > 1e-8 ? (d.x * (c.y - p0.y) - d.y * (c.x - p0.x)) * inversesqrt(L2) : 0.0;
    float u = mix(v_par.x, v_par.y, s);
    float along = mix(v_par.z, v_par.w, s);
    float b = bayer4(ivec2(int(floor(along)), int(floor(side + 64.0))));
    float half_w = 0.5 * u_w;

    vec4 o = vec4(0.0);
    if (u_style == 1) {                       // kontur
        if (dist < half_w) o.r = fade(u, b);
        if (dist < half_w + u_outline) o.g = u_aux.x;
    } else if (u_style == 2) {                // perlen: nur der faden
        if (dist < 0.5) o.r = u_aux.x * (1.0 - 0.4 * u);
    } else if (u_style == 3) {                // lineal
        if (dist < half_w) o.r = fade(u, b);
    } else if (u_style == 4) {                // halbton
        float n = max(u_aux.y, 1.0);
        float q = n > 1.0 ? min(floor(u * n), n - 1.0) / (n - 1.0) : 0.0;
        float cov = 1.0 - (1.0 - u_aux.x) * q;
        if (dist < half_w && b < cov) o.r = 1.0;
    } else {                                  // schiene
        float f = fade(u, b);
        if (dist < half_w + u_outline) o.g = u_aux.y;
        if (dist < half_w) o.b = u_aux.x * f;
        if (dist < 0.5) o.r = f;
    }
    if (o == vec4(0.0)) discard;
    o_buf = o;
}
