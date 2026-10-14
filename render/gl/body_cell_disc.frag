#version 330

// Pixel-pass, grundscheibe eines koerpers (vertex: body.vert, das quad hat
// genau den kernradius). Ohne vektor-zeichnung traegt sie das ganze bild;
// mit zeichnung ist sie der dunkle "grund" zwischen den stufen und damit
// eine eigene region.

in vec2 v_local;

uniform vec3 u_base_color;
uniform vec3 u_light;
uniform float u_light_exp;
uniform float u_emissive;
uniform float u_ground;        // 1 = grund unter der zeichnung, 0 = scheibe allein

// ---- gemeinsam mit body_cell_disc.frag / body_cell_fill.frag ----------
// (GLSL 330 hat kein #include; die beiden kopien muessen gleich bleiben.)
//
// Jedes fragment dieses passes ist EINE ZELLE des koerper-rasters. Es
// schreibt zwei ziele:
//   o_cell = (farbe der stufe, halbton-groesse 0.25 / 0.5 / 0.75 / 1.0)
//   o_id   = (regions-kennung = unbeleuchtete grundfarbe, 1 = belegt)
// Der zusammenbau (body_pixel.frag) formt daraus die zellen, die striche
// zwischen regionen und den rand.

uniform vec2 u_center_px;
uniform vec2 u_viewport;
uniform float u_cell;
uniform float u_seed;
uniform vec2 u_dither_range;   // zelle nur, wenn lo <= bayer < hi

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

// Vier stufen je farbe: schatten kuehl und zum HUD-grund, licht warm und
// zur HUD-schrift. Der farbton-versatz ist, was es GEMALT aussehen laesst.
vec3 ramp(vec3 c, int t) {
    if (t <= 0) return mix(HUD_GROUND * 2.0, c * vec3(0.80, 0.88, 1.10), 0.34);
    if (t == 1) return c * vec3(0.64, 0.70, 0.82);
    if (t == 2) return c;
    return mix(c * vec3(1.08, 1.03, 0.94), HUD_INK, 0.30);
}

// Licht -> stufe. Gedithert wird NUR die grenze tag/nacht (0/1); die
// uebrigen grenzen sind hart.
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

void emit(vec3 base, float lit, float emissive, vec3 region) {
    ivec2 c = local_cell();
    float b = bayer4(c);
    if (b < u_dither_range.x || b >= u_dither_range.y) {
        discard;
    }
    if (emissive >= 0.5) {
        // Stern: helligkeit traegt die randverdunklung der zeichnung, keine
        // nachtseite, kein halbton. Wenige zellen eine stufe tiefer sind
        // die granulation.
        // Eigene, WARME rampe: die kuehle schattenstufe der planeten machte
        // den sonnenrand oliv.
        float l = dot(base, vec3(0.299, 0.587, 0.114));
        int t = l > 0.66 ? 3 : (l > 0.46 ? 2 : 1);
        if (cell_hash(c, u_seed + 3.0) < 0.12) t = max(t - 1, 1);
        vec3 star = t == 3 ? mix(base, vec3(1.0, 0.98, 0.90), 0.55)
                  : (t == 2 ? base * vec3(1.05, 0.80, 0.45)
                            : base * vec3(1.0, 0.60, 0.28));
        o_cell = vec4(min(star, vec3(1.0)), 1.0);
        o_id = vec4(0.5, 0.5, 0.5, 1.0);
        return;
    }
    o_cell = vec4(ramp(base, tone_of(lit, b)), fill_of(lit));
    o_id = vec4(region, 1.0);
}
// ---- ende gemeinsam -------------------------------------------------------

void main() {
    float r = length(v_local);
    if (r > 1.0) {
        discard;
    }
    vec2 p = vec2(v_local.x, -v_local.y);
    float z = sqrt(max(0.0, 1.0 - r * r));
    float lambert = max(dot(vec3(p, z), u_light), 0.0);
    float lit = mix(pow(lambert, u_light_exp), 1.0, u_emissive);
    vec3 base = u_base_color * mix(1.0, 0.5, u_ground);
    if (u_emissive >= 0.5) {
        // Randverdunklung auch ohne zeichnung (kleiner stern).
        base = u_base_color * (1.0 - 0.45 * (1.0 - z));
    }
    emit(base, lit, u_emissive, base);
}
