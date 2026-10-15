#version 330

// Zusammenbau des schiffs aus seinem zellraster, in VOLLER aufloesung --
// dieselbe sprache wie die koerper (`body_pixel.frag`), zwei auspraegungen
// (u_style):
//   2 halbton  licht -> deckung: ab 4-px-zellen ein schrumpfendes quadrat,
//              darunter geordnetes dithern IN der zelle; dunkle naehte
//              zwischen regionen, heller rand auf der lichtseite
//   4 sprite   volle zellen, gedithertes licht ueber alle vier stufen,
//              selout zwischen regionen, 1-px-kontur aussen
// Ausserhalb der zellen: die abgasfahne aus dem hitze-puffer der partikel
// (`ship/plume.py`, `ship_plume.{vert,frag}`), gefaerbt ueber eine
// geditherte glut-rampe aus der akzentfarbe -- weissgluehender kern, gelb,
// akzent, glut, kondensat; der farbton folgt der temperatur der partikel.
// Halbton als punkte, sprite in ganzen zellen.
// Darunter der SCHEIN: die hitze, in zwei gauss-paessen weichgezeichnet
// (`ship_glow.frag`), hier je BILDSCHIRMPIXEL bilinear abgetastet und in
// alpha-stufen mit 1-px-bayer gedithert -- weich im verlauf, hart im pixel.
//
// Koordinaten: GL-pixel (y nach OBEN); u_origin ist die linke UNTERE ecke der
// zelle (0, 0), eine ganze pixelposition. Fuge und eigene kanten liegen wie
// bei den koerpern RECHTS und UNTEN.

uniform sampler2D u_cells;     // farbe + licht (0..1)
uniform sampler2D u_ids;       // regions-kennung + kennzeichen
uniform sampler2D u_heat;      // einheit 2: hitze, kondensat, hitze x temperatur
uniform sampler2D u_glow;      // einheit 3: weichgezeichnete hitze (schein), LINEAR
uniform float u_glow_gain;
uniform float u_glow_temp;     // spool 0..1: farbe des scheins
uniform vec2 u_origin;
// Schiffsmitte relativ zu u_origin, ganze pixel und ein vielfaches von u_k.
// Jedes bayer-muster zaehlt von HIER: u_origin wandert mit der reichweite
// der fahne (zufaellige partikel), und von der ecke aus gezaehlt kippte
// dabei das ganze dithern des rumpfs, auch im leerlauf.
uniform ivec2 u_anchor;
uniform ivec2 u_grid;
uniform int u_k;
uniform int u_style;
uniform vec3 u_light;          // y nach oben
uniform float u_emissive;
uniform vec3 u_grout;
uniform vec3 u_rim;
uniform vec3 u_ink;
uniform vec3 u_accent;
uniform float u_alpha;

out vec4 fragColor;

float bayer4(ivec2 p) {
    int i = (p.x & 3) + 4 * (p.y & 3);
    int m[16] = int[16](0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5);
    return (float(m[i]) + 0.5) / 16.0;
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
// tragen die regionsgrenzen, jede genau einmal.
vec3 edge(ivec2 c, ivec2 d, bool owner, vec4 cell, vec4 id, vec3 col) {
    vec4 nc, ni;
    vec2 lxy = length(u_light.xy) > 0.05 ? normalize(u_light.xy) : vec2(0.0);
    bool lit_side = u_emissive > 0.5 || dot(vec2(d), lxy) > 0.15;
    if (!fetch(c + d, nc, ni)) {
        if (is_accent(id)) return col;
        if (u_style == 4) return lit_side ? min(col * 1.25 + u_ink * 0.25, vec3(1.0))
                                          : col * 0.62;
        return lit_side ? u_rim : col;
    }
    // Holme sind selbst schon eine linie: keine grenze um sie herum.
    if (!owner || is_accent(id) || is_accent(ni) || is_line(id) || is_line(ni)) return col;
    bool diff = distance(id.rgb, ni.rgb) > 0.02 || abs(id.a - ni.a) > 0.05;
    if (!diff) return col;
    if (u_style == 4) return min(cell.rgb, nc.rgb) * 0.55;        // selout
    return mix(u_grout, max(cell.rgb, nc.rgb), 0.25);            // dunkle naht
}

// Glut-rampe: die HITZE waehlt die stufe (helligkeit, deckung), die
// TEMPERATUR der zelle (hitzegewichtet aus den partikeln) den farbton:
// kuehl = tiefes rot und dunkle glut, mittel = akzent/orange, heiss = gelb
// mit weissem bis weiss-blauem kern. Die grenzen sind mit bayer verrauscht,
// damit die stufen als pixel-kanten und nicht als ringe lesen. Im vakuum
// gibt es keinen rauch, nur duennes kaltes kondensat.
vec4 fire(float h, float cold, float T, float b) {
    float hh = h + (b - 0.5) * 0.22;
    float warm = smoothstep(0.25, 0.75, T);
    float hot = smoothstep(0.80, 1.00, T);
    vec3 red = u_accent * vec3(0.78, 0.55, 0.55);
    vec3 orange = mix(u_accent, vec3(1.0, 0.58, 0.24), 0.55);
    vec3 yellow = mix(vec3(1.0, 0.70, 0.30), vec3(1.0, 0.90, 0.55), warm);
    vec3 white = mix(vec3(1.0, 0.95, 0.84), vec3(0.84, 0.92, 1.0), hot);
    if (hh > 1.95) return vec4(mix(yellow, white, 0.35 + 0.65 * warm), 1.0);
    if (hh > 1.15) return vec4(mix(orange, yellow, 0.35 + 0.65 * warm), 1.0);
    if (hh > 0.62) return vec4(mix(red, orange, 0.25 + 0.75 * warm), 1.0);
    if (hh > 0.30) return vec4(mix(red, u_accent, warm), 1.0);
    if (hh > 0.13) return vec4(mix(red * 0.55, u_accent * 0.55, warm), 0.92);
    // Duennes gas am rand der aufweitung: im vakuum laeuft der strahl nicht
    // in einer kante aus, sondern verduennt sich -- eine letzte, schwache
    // und licht gesetzte stufe.
    if (hh > 0.05 && b < 0.6) return vec4(mix(red * 0.40, u_accent * 0.45, warm), 0.65);
    if (cold + (b - 0.5) * 0.10 > 0.09) {
        // Kondensat: kuehl und duenn, im schein der glut leicht angewaermt.
        vec3 c = mix(u_grout, u_ink * vec3(0.72, 0.82, 0.92), 0.20);
        vec3 lit = mix(red * 0.6, vec3(1.0, 0.75, 0.45), 0.25);
        return vec4(mix(c, lit, clamp(h * 6.0, 0.0, 0.7)), 0.32 + 0.25 * clamp(h * 6.0, 0.0, 1.0));
    }
    return vec4(0.0);
}

void main() {
    ivec2 rel = ivec2(floor(gl_FragCoord.xy - u_origin));
    int k = u_k;
    ivec2 c = cell_of(rel);
    ivec2 sub = rel - c * k;
    ivec2 q = rel - u_anchor;          // pixel ab schiffsmitte
    ivec2 qc = c - u_anchor / k;       // zelle ab schiffsmitte

    vec4 cell, id;
    if (fetch(c, cell, id)) {
        float lit = cell.a;
        bool acc = is_accent(id);
        vec3 col = cell.rgb;

        if (u_style == 2) {
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
                ink = bayer4(q) < f;
            }
            if (!ink) {
                fragColor = out_col(u_grout, 0.55);
                return;
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

    vec4 hq = (c.x >= 0 && c.y >= 0 && c.x < u_grid.x && c.y < u_grid.y)
            ? texelFetch(u_heat, c, 0) : vec4(0.0);
    vec2 hs = hq.rg;
    float temp = hq.r > 0.005 ? hq.b / hq.r : 0.5;

    // Schein an diesem pixel: stetig zwischen den zellmitten, dann in
    // stufen von 0.10 alpha gedithert (je pixel, nicht je zelle).
    vec2 guv = (gl_FragCoord.xy - u_origin) / float(k) / vec2(textureSize(u_glow, 0));
    float g = texture(u_glow, guv).r * u_glow_gain;
    float gb = bayer4(q);
    float ga = clamp(floor(g * 0.80 / 0.10 + gb) * 0.10, 0.0, 0.70);
    // Farbe des scheins: innen heller, und mit dem schub waermer bis
    // weiss-blau (heisseres gas), bei wenig schub ein dunkles rot.
    vec3 gin = mix(vec3(1.0, 0.70, 0.36), vec3(0.88, 0.92, 1.0), u_glow_temp * u_glow_temp * 0.6);
    vec3 gout = mix(u_accent * vec3(0.85, 0.6, 0.6), u_accent, u_glow_temp);
    vec3 gc = mix(gout, gin, clamp(g * 0.9, 0.0, 0.75) * (0.45 + 0.55 * u_glow_temp));

    if (hs.x > 0.02 || hs.y > 0.02) {
        vec4 f = fire(hs.x, hs.y, temp, bayer4(qc));
        if (f.a > 0.0) {
            if (u_style == 4) {
                // Kondensat (halb durchsichtig) liegt UEBER dem schein.
                fragColor = out_col(mix(gc, f.rgb, f.a < 1.0 && ga > 0.0 ? f.a : 1.0),
                                    max(f.a, ga));
                return;
            }
            // Halbton: deckung aus der hitze, wie beim rumpf (ab 4-px-zellen
            // ein quadrat, darunter dithern in der zelle).
            float cov = clamp(max(hs.x * 0.85, hs.y * 1.6), 0.25, 1.0);
            bool ink;
            if (k >= 4) {
                int n = clamp(int(floor(float(k) * sqrt(cov) + 0.5)), 1, k);
                int off = (k - n) / 2;
                ivec2 p = sub - ivec2(off, k - n - off);
                ink = p.x >= 0 && p.y >= 0 && p.x < n && p.y < n;
            } else {
                ink = bayer4(q) < cov;
            }
            if (ink) {
                fragColor = out_col(mix(gc, f.rgb, f.a < 1.0 && ga > 0.0 ? f.a : 1.0),
                                    max(f.a, ga));
                return;
            }
        }
    }
    if (ga > 0.0) {
        fragColor = out_col(gc, ga);
        return;
    }
    discard;
}
