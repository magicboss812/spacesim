#version 330

// Eine marke in den puffer (blending MAX), pixelgenau ueber die pixelmitte.
//   perlen  achsparallele zelle (size x size, gross: size + 1) mit 1-px-ring
//   lineal  strich quer zur bahn, 1 px breit (klein: eine seite, gross: beide)
//   schiene pfeilspitze in flugrichtung, zwei 45-grad-schenkel, dunkler ring
// Kanaele wie `pred_seg.frag`: r = linie, g = saum, a = marke.

flat in vec4 v_a;
flat in vec4 v_b;

uniform int u_style;
uniform vec3 u_m;          // perlen: size, ring; lineal: strich, langer strich; schiene: pfeil, ring
uniform vec2 u_fade;

out vec4 o_buf;

float fade(float u) {
    float n = u_fade.x;
    if (n <= 1.0) return 1.0;
    float q = clamp(floor(u * n), 0.0, n - 1.0);
    return 1.0 - (1.0 - u_fade.y) * q / (n - 1.0);
}

void main() {
    vec4 o = vec4(0.0);
    float u = v_b.x;
    bool major = v_b.y > 0.5;
    if (u_style == 2) {
        int size = int(u_m.x + 0.5) + (major ? 1 : 0);
        ivec2 q = ivec2(floor(gl_FragCoord.xy)) - ivec2(floor(v_a.xy));
        int lo = -(size / 2);
        int hi = lo + size - 1;
        bool core = q.x >= lo && q.x <= hi && q.y >= lo && q.y <= hi;
        bool ring = q.x >= lo - 1 && q.x <= hi + 1 && q.y >= lo - 1 && q.y <= hi + 1;
        if (ring) o.g = major ? 0.85 : u_m.y;
        if (core) {
            if (major) o.a = 1.0;
            // Perlen verblassen nur zu 40 % so weit wie die linie: sie
            // tragen die bahn, der faden allein waere zu leicht zu verlieren.
            else o.r = mix(1.0, fade(u), 0.4);
        }
    } else {
        vec2 p = floor(gl_FragCoord.xy) + 0.5 - v_a.xy;
        vec2 t = v_a.zw;
        vec2 n = vec2(-t.y, t.x);
        float lx = dot(p, t);
        float ly = dot(p, n);
        if (u_style == 3) {
            bool hit = major ? (abs(lx) < 0.5 && abs(ly) < u_m.y + 0.5)
                             : (abs(lx) < 0.5 && ly > 0.5 && ly < u_m.x + 0.5);
            if (hit) {
                if (major) o.a = fade(u);
                else o.r = fade(u);
            }
        } else {
            // Schenkel |ly| = 0.6 * back: rund 31 grad zur bahn. Mit 45 grad
            // stand auf einer diagonalen bahn ein achsparalleler winkel da,
            // der nicht mehr als pfeil las.
            float back = -lx;
            float e = abs(abs(ly) - 0.6 * back) * 0.857;   // 1/sqrt(1 + 0.6^2)
            if (back > -1.5 && back < u_m.x + 1.5 && e < 1.5) o.g = u_m.y;
            if (back > -0.5 && back < u_m.x + 0.5 && e < 0.5) o.a = fade(u);
        }
    }
    if (o == vec4(0.0)) discard;
    o_buf = o;
}
