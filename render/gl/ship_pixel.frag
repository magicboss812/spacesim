#version 330

// Zusammenbau des schiffs aus seinem zellraster, in VOLLER aufloesung --
// dieselbe sprache wie die koerper (`body_pixel.frag`), fuenf auspraegungen
// (u_style):
//   1 LED       gefaste zellen, fuge auf der lichtseite und am rand,
//               1-px-striche zwischen regionen, heller lichtrand
//   2 halbton   die zelle schrumpft mit dem licht, nachtseite als punkte
//   3 blaupause dunkle paneele, jede regionsgrenze ein strich in HUD-schrift
//   4 sprite    volle zellen, gedithertes licht, dunkle 1-px-kontur aussen
//   5 signal    einfarbig in HUD-schrift, jede zelle eine LED (punktmatrix)
// Ausserhalb der zellen: die abgasfahne als punktmatrix (bei 4 in stufen).
//
// Koordinaten: GL-pixel (y nach OBEN); u_origin ist die linke UNTERE ecke der
// zelle (0, 0), eine ganze pixelposition. Fuge und eigene kanten liegen wie
// bei den koerpern RECHTS und UNTEN.

uniform sampler2D u_cells;     // farbe + licht (0..1)
uniform sampler2D u_ids;       // regions-kennung + kennzeichen
uniform vec2 u_origin;
uniform ivec2 u_grid;
uniform int u_k;
uniform int u_style;
uniform vec3 u_light;          // y nach oben
uniform float u_emissive;
uniform vec3 u_grout;
uniform vec3 u_rim;
uniform vec3 u_ink;
uniform vec3 u_accent;
uniform float u_plume;         // 0..1
uniform vec2 u_center;         // schiffsmitte, GL-pixel
uniform vec2 u_head;
uniform float u_scale;
uniform float u_alpha;

out vec4 fragColor;

float bayer4(ivec2 p) {
    int i = (p.x & 3) + 4 * (p.y & 3);
    int m[16] = int[16](0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5);
    return (float(m[i]) + 0.5) / 16.0;
}

float luma(vec3 c) {
    return dot(c, vec3(0.299, 0.587, 0.114));
}

bool fetch(ivec2 c, out vec4 cell, out vec4 id) {
    if (c.x < 0 || c.y < 0 || c.x >= u_grid.x || c.y >= u_grid.y) {
        cell = vec4(0.0);
        id = vec4(0.0);
        return false;
    }
    cell = texelFetch(u_cells, c, 0);
    id = texelFetch(u_ids, c, 0);
    return id.a > 0.1;
}

bool occupied(ivec2 c) {
    vec4 a, b;
    return fetch(c, a, b);
}

ivec2 cell_of(ivec2 rel) {
    return ivec2(floor(vec2(rel) / float(u_k)));
}

vec4 out_col(vec3 c, float a) {
    return vec4(c, a * u_alpha);
}

bool is_accent(vec4 id) { return abs(id.a - 0.85) < 0.05; }
bool is_line(vec4 id) { return abs(id.a - 0.7) < 0.05; }

// Kante der zelle c in richtung d (y oben). owner: rechte und untere kante
// tragen die regionsgrenzen, jede genau einmal. Die grenzen tragen die
// zeichnung: die panelfarben des mockups liegen dicht beieinander, seine
// identitaet sind die hellen konturen auf dunklen flaechen.
vec3 edge(ivec2 c, ivec2 d, bool owner, vec4 cell, vec4 id, vec3 col) {
    vec4 nc, ni;
    vec2 lxy = length(u_light.xy) > 0.05 ? normalize(u_light.xy) : vec2(0.0);
    bool lit_side = u_emissive > 0.5 || dot(vec2(d), lxy) > 0.15;
    if (!fetch(c + d, nc, ni)) {
        if (is_accent(id)) return col;
        if (u_style == 1) return lit_side ? u_rim : mix(col, u_ink, 0.40);
        if (u_style == 2) return lit_side ? u_rim : col;
        if (u_style == 3) return lit_side ? u_ink : mix(u_ink, u_grout, 0.45);
        if (u_style == 4) return lit_side ? min(col * 1.25 + u_ink * 0.25, vec3(1.0))
                                          : col * 0.62;
        return lit_side ? u_ink : col;
    }
    // Holme sind selbst schon eine linie: keine grenze um sie herum.
    if (!owner || is_accent(id) || is_accent(ni) || is_line(id) || is_line(ni)) return col;
    bool diff = distance(id.rgb, ni.rgb) > 0.02 || abs(id.a - ni.a) > 0.05;
    if (!diff) return col;
    float l = max(cell.a, nc.a);
    vec3 hi = max(cell.rgb, nc.rgb);
    if (u_style == 1) return mix(hi, u_ink, 0.30) * mix(0.62, 1.0, smoothstep(0.4, 0.95, l));
    if (u_style == 2) return mix(u_grout, hi, 0.25);              // dunkle naht
    if (u_style == 3) return mix(mix(u_ink, u_grout, 0.55), u_ink, smoothstep(0.4, 0.95, l));
    if (u_style == 4) return min(cell.rgb, nc.rgb) * 0.55;        // selout
    return mix(hi, u_ink, 0.45);
}

// Abgasfahne: im vektor-mockup ineinander liegende ellipsen, zur duesen-
// spitze (anker) geschrumpft. s = auf welcher schicht der punkt liegt
// (0 am anker, 1 am aeusseren rand) -- geschlossen geloest.
float plume_layer(vec2 p) {
    vec2 q = p - u_center;
    vec2 loc = vec2(dot(q, u_head), dot(q, vec2(-u_head.y, u_head.x))) / max(u_scale, 1e-6);
    vec2 u = loc - vec2(-81.0, 0.0);
    float X = u.x / 32.0;
    float Y = u.y / 21.0;
    if (X >= 0.0) return 2.0;
    return -(X * X + Y * Y) / (2.0 * X);
}

void main() {
    ivec2 rel = ivec2(floor(gl_FragCoord.xy - u_origin));
    int k = u_k;
    ivec2 c = cell_of(rel);
    ivec2 sub = rel - c * k;
    int s = max(k - 1, 1);
    // Feld der zelle ohne fuge (rechts, unten): q in 0..s-1, y nach oben.
    ivec2 q = ivec2(sub.x, sub.y - 1);
    bool in_field = k < 2 || (sub.x < s && sub.y >= 1);

    vec4 cell, id;
    if (fetch(c, cell, id)) {
        float lit = cell.a;
        bool acc = is_accent(id);
        bool boundary = !occupied(c + ivec2(1, 0)) || !occupied(c + ivec2(-1, 0))
                     || !occupied(c + ivec2(0, 1)) || !occupied(c + ivec2(0, -1));
        vec3 grout = mix(u_grout, cell.rgb, 0.40);
        vec3 col = cell.rgb;

        if (u_style == 1 || u_style == 5) {
            // LED: gefastes quadrat. Bei 1 nur lichtseite und rand, bei 5
            // ueberall (punktmatrix-anzeige).
            // Bei 1 erst ab 4-px-zellen: kleiner bleibt vom gefasten quadrat
            // nur ein gitter aus fugen, das den rumpf dunkel schraffiert.
            float gap = u_style == 5 ? 1.0
                      : (k >= 4 ? max(smoothstep(0.70, 0.92, lit), boundary ? 1.0 : 0.0) : 0.0);
            if (acc && k >= 4) gap = 1.0;
            bool ink = in_field;
            if (ink && s >= 3 && gap > 0.25) {
                int m = s - 1;
                if (q.x + q.y < 1 || (m - q.x) + q.y < 1
                        || q.x + (m - q.y) < 1 || (m - q.x) + (m - q.y) < 1) {
                    ink = false;
                }
            }
            col = ink ? cell.rgb : mix(cell.rgb, grout, gap);
        } else if (u_style == 2) {
            // Halbton: deckung aus dem licht. Ab 4-px-zellen ein
            // schrumpfendes quadrat; kleiner gaebe es nur zwei groessen (voll
            // oder 1 px), daher dort geordnetes dithern INNERHALB der zelle.
            float f = acc ? 1.0 : clamp((lit - 0.30) / 0.60, 0.25, 1.0);
            bool ink;
            if (k >= 4) {
                int n = clamp(int(floor(float(k) * sqrt(f) + 0.5)), 1, k);
                int off = (k - n) / 2;
                ivec2 p = sub - ivec2(off, k - n - off);
                ink = p.x >= 0 && p.y >= 0 && p.x < n && p.y < n;
            } else {
                ink = bayer4(rel) < f;
            }
            if (!ink) {
                fragColor = out_col(u_grout, 0.55);
                return;
            }
        } else if (u_style == 3) {
            // Blaupause: das paneel dunkel, die farbe nur als hauch.
            col = acc ? cell.rgb : mix(u_grout, cell.rgb, 0.22 + 0.18 * lit);
            if (is_line(id)) {
                col = mix(col, u_ink, 0.45);     // konturzellen als linie
            }
        }

        if (sub.x == k - 1) col = edge(c, ivec2(1, 0), true, cell, id, col);
        else if (sub.x == 0) col = edge(c, ivec2(-1, 0), false, cell, id, col);
        if (sub.y == 0) col = edge(c, ivec2(0, -1), true, cell, id, col);
        else if (sub.y == k - 1) col = edge(c, ivec2(0, 1), false, cell, id, col);
        fragColor = out_col(col, 1.0);
        return;
    }

    // --- ausserhalb der zellen ---------------------------------------------
    if (u_style == 4) {
        // Kontur, genau 1 px, rund um die silhouette -- eine stufe heller
        // als der grund, sonst verschwaende sie auf dem dunklen hintergrund.
        if (occupied(cell_of(rel + ivec2(1, 0))) || occupied(cell_of(rel - ivec2(1, 0)))
                || occupied(cell_of(rel + ivec2(0, 1))) || occupied(cell_of(rel - ivec2(0, 1)))) {
            fragColor = out_col(mix(u_grout, u_ink, 0.16), 1.0);
            return;
        }
    }

    if (u_plume > 0.004) {
        vec2 cc = (vec2(c) + 0.5) * float(k) + u_origin;
        float layer = plume_layer(cc);
        if (layer < 1.0) {
            float dens = (1.0 - layer) * u_plume;
            vec3 hot = mix(u_accent, vec3(1.0, 0.95, 0.85), 0.55);
            if (u_style == 4) {
                // Gestuft: kern, akzent, dunkler saum -- ganze zellen.
                float b = (bayer4(c) - 0.5) * 0.12;
                if (dens > 0.62 + b) { fragColor = out_col(hot, 1.0); return; }
                if (dens > 0.36 + b) { fragColor = out_col(u_accent, 1.0); return; }
                if (dens > 0.16 + b) { fragColor = out_col(u_accent * 0.45, 1.0); return; }
                discard;
            }
            int n = u_style == 2 ? clamp(int(ceil(dens * float(s))), 1, s)
                  : (dens > 0.55 ? s : max(s - 1, 1));
            int off = (s - n) / 2;
            ivec2 p = q - ivec2(off);
            bool dot_ok = in_field && p.x >= 0 && p.y >= 0 && p.x < n && p.y < n;
            float thresh = u_style == 2 ? 0.08 : bayer4(c);
            if (dot_ok && dens > thresh) {
                vec3 pc = mix(u_accent, hot, smoothstep(0.5, 0.9, dens));
                if (u_style == 3) pc = mix(u_accent, u_ink, 0.25);
                fragColor = out_col(pc, 0.95);
                return;
            }
        }
    }
    discard;
}
