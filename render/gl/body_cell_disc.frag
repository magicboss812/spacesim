#version 330

// Pixel-pass, ganze scheibe eines koerpers (vertex: body.vert, quad = kern).
// Drei arten:
//  0 planet  -- grund unter der vektor-zeichnung bzw. die scheibe allein;
//  1 stern   -- prozedural je zelle: randverdunklung in ringen, granulation
//               als zusammenhaengende zellen (voronoi auf der kugel);
//  2 gasriese -- prozedural je zelle: breitenbaender aus denselben zahlen wie
//               die vektor-zeichnung (`bodies/style.py::gas_params`).
// Stern und gasriese brauchen damit KEINE detailstufen.

in vec2 v_local;

uniform vec3 u_base_color;
uniform float u_ground;        // 1 = grund unter der zeichnung
uniform int u_kind;

uniform mat3 u_gas_rot;        // bildraum -> musterraum ist transpose
uniform vec3 u_gas_axis;
uniform vec4 u_gas;            // dichte, phase, wirbel, frequenz
uniform int u_gas_tiers[24];
uniform int u_gas_band_min;
uniform vec3 u_gas_col[3];

// ---- gemeinsam mit body_cell_disc.frag / body_cell_fill.frag ----------
// (GLSL 330 hat kein #include; die beiden kopien muessen gleich bleiben.)
//
// Jedes fragment dieses passes ist EINE ZELLE des koerper-rasters. Es
// schreibt zwei ziele:
//   o_cell = (farbe der stufe, halbton-groesse 0.25 / 0.5 / 0.75 / 1.0)
//   o_id   = (regions-kennung = unbeleuchtete grundfarbe, kennzeichen)
// Kennzeichen: 1.0 = zeichnung bzw. prozedurale flaeche, 0.4 = grund ohne
// grenzstriche. Der zusammenbau (body_pixel.frag) zieht striche nur
// zwischen gleichen kennzeichen.

uniform vec2 u_center_px;
uniform vec2 u_viewport;
uniform float u_cell;
uniform float u_seed;
uniform float u_tag;
uniform float u_body_radius;   // kernradius in pixeln
uniform vec3 u_light;
uniform float u_light_exp;
uniform float u_emissive;

layout(location = 0) out vec4 o_cell;
layout(location = 1) out vec4 o_id;

// HUD-grund (#04070c) und HUD-schrift (#dceaf4): die rampe jedes koerpers
// laeuft an beiden enden in die farben des interfaces.
const vec3 HUD_GROUND = vec3(0.016, 0.027, 0.047);
const vec3 HUD_INK = vec3(0.863, 0.918, 0.957);

float bayer4(ivec2 p) {
    int i = (p.x & 3) + 4 * (p.y & 3);
    int m[16] = int[16](0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5);
    return (float(m[i]) + 0.5) / 16.0;
}

float cell_hash(ivec2 p, float seed) {
    uint h = uint(p.x) * 374761393u + uint(p.y) * 668265263u
             + uint(seed) * 2246822519u;
    h = (h ^ (h >> 13u)) * 1274126177u;
    h ^= h >> 16u;
    return float(h & 0xFFFFu) / 65535.0;
}

// Zelle relativ zur koerpermitte: haengt nicht am schirm, also wandert beim
// schwenken nichts ueber den koerper.
ivec2 local_cell() {
    vec2 ref = vec2(u_center_px.x, u_viewport.y - u_center_px.y) / u_cell;
    return ivec2(floor(gl_FragCoord.xy)) - ivec2(floor(ref + 0.5));
}

// Kugelpunkt unter der ZELLMITTE (y nach oben). Die beleuchtung haengt
// daran und nicht an der facette: flach beleuchtete facetten lagen als
// dreiecke aus stufen ueber der nachtseite.
vec3 cell_normal() {
    vec2 cpx = vec2((floor(gl_FragCoord.x) + 0.5) * u_cell,
                    u_viewport.y - (floor(gl_FragCoord.y) + 0.5) * u_cell);
    vec2 p = (cpx - u_center_px) / max(u_body_radius, 1e-3);
    p.y = -p.y;
    float rr = min(dot(p, p), 1.0);
    return vec3(p, sqrt(1.0 - rr));
}

float cell_lit(vec3 n) {
    float d = max(dot(n, u_light), 0.0);
    return mix(pow(d, u_light_exp), 1.0, u_emissive);
}

// Vier stufen je farbe: schatten kuehl und zum HUD-grund, licht warm und
// zur HUD-schrift.
vec3 ramp(vec3 c, int t) {
    if (t <= 0) return mix(HUD_GROUND * 2.0, c * vec3(0.80, 0.88, 1.10), 0.34);
    if (t == 1) return c * vec3(0.64, 0.70, 0.82);
    if (t == 2) return c;
    // Licht: in der eigenen farbe heller und waermer, nur ein hauch zur
    // HUD-schrift -- staerker gemischt wurde jede farbe zu grau.
    return min(c * vec3(1.30, 1.24, 1.10) + HUD_INK * 0.10, vec3(1.0));
}

// Licht -> stufe. Gedithert wird NUR die grenze tag/nacht (0/1).
int tone_of(float lit, float b) {
    if (lit < 0.07 + (b - 0.5) * 0.10) return 0;
    if (lit < 0.30) return 1;
    if (lit < 0.68) return 2;
    return 3;
}

// Licht -> halbton: am terminator schrumpft die zelle, auf der nachtseite
// bleibt ein punkt, damit die form des koerpers stehen bleibt.
float fill_of(float lit) {
    if (lit > 0.18) return 1.0;
    if (lit > 0.08) return 0.75;
    if (lit > 0.03) return 0.5;
    return 0.25;
}

void emit_lit(vec3 base, vec3 region, float tag) {
    ivec2 c = local_cell();
    float lit = cell_lit(cell_normal());
    o_cell = vec4(ramp(base, tone_of(lit, bayer4(c))), fill_of(lit));
    o_id = vec4(region, tag);
}
// ---- ende gemeinsam -------------------------------------------------------

float hash31(vec3 p) {
    p = fract(p * 0.1031 + u_seed * 0.0001);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}

float vnoise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = mix(mix(hash31(i), hash31(i + vec3(1, 0, 0)), f.x),
                  mix(hash31(i + vec3(0, 1, 0)), hash31(i + vec3(1, 1, 0)), f.x), f.y);
    float b = mix(mix(hash31(i + vec3(0, 0, 1)), hash31(i + vec3(1, 0, 1)), f.x),
                  mix(hash31(i + vec3(0, 1, 1)), hash31(i + vec3(1, 1, 1)), f.x), f.y);
    return mix(a, b, f.z);
}

float fbm(vec3 p) {
    return 0.5 * vnoise(p) + 0.3 * vnoise(p * 2.03 + 11.0) + 0.2 * vnoise(p * 4.1 + 23.0);
}

// Abstand zur naechsten und zweitnaechsten zufallszelle (voronoi, 3D).
vec2 worley(vec3 p) {
    vec3 i = floor(p);
    float f1 = 9.0, f2 = 9.0;
    for (int z = -1; z <= 1; ++z)
    for (int y = -1; y <= 1; ++y)
    for (int x = -1; x <= 1; ++x) {
        vec3 g = i + vec3(x, y, z);
        vec3 o = vec3(hash31(g), hash31(g + 17.0), hash31(g + 41.0));
        float d = length(g + o - p);
        if (d < f1) { f2 = f1; f1 = d; } else if (d < f2) { f2 = d; }
    }
    return vec2(f1, f2);
}

void main() {
    if (length(v_local) > 1.0) {
        discard;
    }
    ivec2 c = local_cell();
    vec3 n = cell_normal();

    if (u_kind == 1) {
        // Stern: vier ringe aus der randverdunklung, granulation als
        // zusammenhaengende zellen. Die koerner haengen an der KUGEL
        // (wachsen beim zoomen mit) und kommen erst, wenn eines mindestens
        // ~3 zellen gross ist -- darunter waeren sie nur rauschen.
        float mu = n.z;
        float I = 1.0 - 0.62 * (1.0 - mu);
        int t = I > 0.90 ? 3 : (I > 0.76 ? 2 : (I > 0.58 ? 1 : 0));
        float grains = 10.0;
        float grain_px = u_body_radius / grains;
        float amp = smoothstep(2.0 * u_cell, 4.0 * u_cell, grain_px);
        vec2 w = worley(n * grains + vec3(u_seed * 0.013));
        if (amp > 0.0 && (w.y - w.x) < 0.16 * amp) {
            t = max(t - 1, 0);
        }
        vec3 base = u_base_color;
        vec3 col = t == 3 ? mix(base, vec3(1.0, 0.98, 0.90), 0.60)
                 : (t == 2 ? base * vec3(1.05, 0.86, 0.52)
                 : (t == 1 ? base * vec3(1.0, 0.62, 0.28)
                           : base * vec3(0.78, 0.36, 0.14)));
        o_cell = vec4(min(col, vec3(1.0)), 1.0);
        o_id = vec4(0.5, 0.5, 0.5, 0.4);
        return;
    }

    if (u_kind == 2) {
        // Gasriese: breite gegen die achse, band = floor(breite*dichte +
        // phase + wirbel). Der wirbel kommt aus dem musterraum (an der
        // kugel), dazu ein feiner anteil je zelle -- zerfranste
        // bandkanten in ganzen zellen.
        vec3 m = transpose(u_gas_rot) * n;
        float lat = asin(clamp(dot(n, u_gas_axis), -1.0, 1.0)) / 1.5707963;
        float turb = fbm(vec3(m.x * u_gas.w + 3.0, m.y * u_gas.w * 2.6, m.z * u_gas.w + 7.0)) - 0.5;
        float fine = (vnoise(m * 22.0 + 5.0) - 0.5) * 0.22;
        int band = int(floor(lat * u_gas.x + u_gas.y + turb * u_gas.z * 1.6 + fine));
        int idx = clamp(band - u_gas_band_min, 0, 23);
        vec3 col = u_gas_col[u_gas_tiers[idx]];
        emit_lit(col, col, 1.0);
        return;
    }

    // Planet: der grund (unter der zeichnung dunkler, eigene region).
    vec3 base = u_base_color * mix(1.0, 0.5, u_ground);
    emit_lit(base, base, 0.4);
}
