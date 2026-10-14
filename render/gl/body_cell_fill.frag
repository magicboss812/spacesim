#version 330

// Pixel-pass, fuellungen der vektor-zeichnung (vertex: body_surface.vert).
// Deckend: die farbe der stufe ist eine WAHL aus der rampe, kein
// durchscheinender schleier. Das licht kommt aus der zellmitte auf der
// kugel, nicht aus der facette (siehe cell_normal).

in vec3 v_color;
in float v_alpha;

// ---- gemeinsam mit body_cell_disc.frag / body_cell_fill.frag ----------
// (GLSL 330 hat kein #include; die beiden kopien muessen gleich bleiben.)
//
// Jedes fragment dieses passes ist EINE ZELLE des koerper-rasters. Es
// schreibt zwei ziele:
//   o_cell = (farbe der stufe, halbton-groesse 0.25 / 0.5 / 0.75 / 1.0)
//   o_id   = (regions-kennung = unbeleuchtete grundfarbe, kennzeichen)
// Kennzeichen: 1.0 / 0.8 = zeichnung (die zwei stufen einer detail-
// ueberblendung), 0.4 = grund ohne grenzstriche. Der zusammenbau
// (body_pixel.frag) zieht striche nur zwischen gleichen kennzeichen -- eine
// ueberblendung erzeugt so keine striche um jede gewechselte zelle.

uniform vec2 u_center_px;
uniform vec2 u_viewport;
uniform float u_cell;
uniform float u_seed;
uniform vec2 u_dither_range;   // zelle nur, wenn lo <= bayer < hi
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

// Ueberblendungen (einblendung der zeichnung, detailstufe) sind ein
// RADIALER WISCH: der schluessel ist der abstand der zellmitte zur
// koerpermitte, der neuere zustand waechst von innen nach aussen mit einer
// sauberen zellgrenze. Ein bayer-dissolve zweier verschiedener muster war in
// der mitte des uebergangs reines rauschen.
bool dither_skip(ivec2 c) {
    vec3 n = cell_normal();
    float key = min(length(n.xy), 0.9999);
    return key < u_dither_range.x || key >= u_dither_range.y;
}

void emit_lit(vec3 base, vec3 region, float tag) {
    ivec2 c = local_cell();
    float lit = cell_lit(cell_normal());
    o_cell = vec4(ramp(base, tone_of(lit, bayer4(c))), fill_of(lit));
    o_id = vec4(region, tag);
}
// ---- ende gemeinsam -------------------------------------------------------

void main() {
    // Eine fuellung ohne deckkraft (ausgeblendete figur) belegt keine zelle.
    if (v_alpha <= 0.002) {
        discard;
    }
    if (dither_skip(local_cell())) {
        discard;
    }
    emit_lit(v_color, v_color, u_tag);
}
