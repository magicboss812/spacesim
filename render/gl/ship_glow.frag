#version 330

// Schein der abgasfahne: eine richtung eines getrennten gauss-filters ueber
// den hitze-puffer im zellraster (vertex: body_pixel.vert ueber den ganzen
// puffer). Zweimal laufen lassen, waagerecht und senkrecht; der
// zusammenbau (`ship_pixel.frag`) tastet das ergebnis dann je
// BILDSCHIRMPIXEL bilinear ab und dithert es in stufen.
//
// Je abtastung ZWEI texel: die quelle ist LINEAR gefiltert, und ein
// abgriff genau zwischen zwei texeln liefert ihr mittel -- die haelfte der
// zugriffe bei (fast) demselben kern.

uniform sampler2D u_src;   // r = hitze, LINEAR
uniform ivec2 u_grid;
uniform ivec2 u_dir;
uniform float u_sigma;     // in zellen

out vec4 o_glow;

void main() {
    vec2 size = vec2(textureSize(u_src, 0));
    vec2 base = floor(gl_FragCoord.xy) + 0.5;
    vec2 dir = vec2(u_dir);
    vec2 lo = vec2(0.5);
    vec2 hi = vec2(u_grid) - 0.5;
    int R = min(int(ceil(u_sigma * 2.5)), 32);
    float inv = 0.5 / max(u_sigma * u_sigma, 1e-3);
    float acc = texture(u_src, base / size).r;
    float wsum = 1.0;
    for (int i = 1; i <= R; i += 2) {
        float w = exp(-float(i * i) * inv) + exp(-float((i + 1) * (i + 1)) * inv);
        float t = float(i) + 0.5;
        for (int sgn = -1; sgn <= 1; sgn += 2) {
            vec2 p = base + dir * t * float(sgn);
            wsum += w;
            if (any(lessThan(p, lo)) || any(greaterThan(p, hi))) {
                continue;
            }
            acc += w * texture(u_src, p / size).r;
        }
    }
    o_glow = vec4(acc / wsum, 0.0, 0.0, 0.0);
}
