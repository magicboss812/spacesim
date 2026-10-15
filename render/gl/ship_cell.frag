#version 330

// Zellpass des schiffs: je zelle eine deckende farbe aus der rampe und das
// licht (o_cell), dazu regions-kennung und kennzeichen (o_id) -- dasselbe
// format wie `body_cell_*.frag`, damit der putz-pass `body_cell_clean.frag`
// unveraendert mitlaeuft.
//
// Licht: der rumpf ist ein ZYLINDER um die laengsachse (|y| <= HULL), die
// radiatoren sind flache paneele, leicht nach aussen gekippt. Die normale
// wird mit der nase gedreht, das licht kommt in bildschirmkoordinaten
// (y oben), wie bei den koerpern.

in vec2 v_local;
in vec3 v_col;
in float v_tag;

uniform vec2 u_head;
uniform vec3 u_light;
uniform float u_light_exp;
uniform float u_emissive;
uniform int u_style;     // 2 halbton, 4 sprite
// Die fahne als lichtquelle (`ship/plume.py::light_sources`): punktlichter
// im schiffszentrierten, nicht mitdrehenden raum (x, y, staerke, radius).
uniform vec4 u_plights[8];
uniform int u_plight_n;
uniform vec3 u_glow;     // farbe dieses lichts
// Schiffsmitte im puffer (wie im vertex-shader). Das bayer-muster zaehlt
// von der mitte aus, nicht von der pufferecke: die ecke wandert mit der
// reichweite der fahne, und jede verschiebung warf das ganze muster um.
uniform vec2 u_center;
uniform float u_cell;

layout(location = 0) out vec4 o_cell;
layout(location = 1) out vec4 o_id;

const float HULL = 17.0;
const vec3 HUD_GROUND = vec3(0.016, 0.027, 0.047);
const vec3 HUD_INK = vec3(0.863, 0.918, 0.957);

float bayer4(ivec2 p) {
    int i = (p.x & 3) + 4 * (p.y & 3);
    int m[16] = int[16](0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5);
    return (float(m[i]) + 0.5) / 16.0;
}

// Dieselbe rampe wie die koerper (`body_cell_disc.frag::ramp`).
vec3 ramp(vec3 c, int t) {
    if (t <= 0) return mix(HUD_GROUND * 2.0, c * vec3(0.80, 0.88, 1.10), 0.34);
    if (t == 1) return c * vec3(0.64, 0.70, 0.82);
    if (t == 2) return c;
    return min(c * vec3(1.30, 1.24, 1.10) + HUD_INK * 0.10, vec3(1.0));
}

void main() {
    ivec2 c = ivec2(floor(gl_FragCoord.xy));
    vec3 base = v_col;
    if (abs(v_tag - 0.85) < 0.05) {
        // Akzent: leuchtet selbst, immer die hellste stufe, voller punkt.
        o_cell = vec4(min(v_col * 1.15 + 0.04, vec3(1.0)), 1.0);
        o_id = vec4(v_col, v_tag);
        return;
    }

    float ly = v_local.y;
    vec3 nl = abs(ly) <= HULL
        ? vec3(0.0, ly / HULL, sqrt(max(0.0, 1.0 - (ly / HULL) * (ly / HULL))))
        : vec3(0.0, sign(ly) * 0.35, 0.937);
    vec3 n = vec3(u_head.x * nl.x - u_head.y * nl.y,
                  u_head.y * nl.x + u_head.x * nl.y, nl.z);
    // Das licht steiler zum betrachter als bei den koerpern: von oben auf
    // ein flaches schiff gesehen kaeme es sonst meist streifend, und der
    // rumpf laege ganz im schatten. Dazu ein grundlicht: das schiff ist
    // spielfigur und verschwindet nie (die koerper haben dafuer ihren
    // halbton-punkt). Die richtung liest man am hellen rand ab.
    vec3 l = normalize(vec3(u_light.xy, max(u_light.z, 0.75)));
    float d = max(dot(n, l), 0.0);
    float lit = mix(0.40 + 0.60 * pow(d, u_light_exp), 1.0, u_emissive);

    // Panelfarben des mockups sind dunkel (#2f3550 ...): fuer die rampe eine
    // stufe angehoben, sonst laege der ganze rumpf unter der schattenstufe.
    vec3 lifted = min(base * 1.55 + 0.03, vec3(1.0));
    float b = bayer4(c - ivec2(floor(u_center / u_cell + 0.5)));
    int t;
    if (u_style == 4) {
        // Sprite: jede stufengrenze gedithert, ein verlauf ueber den
        // zylinder, aus dem ROHEN licht (ohne grundlicht) fuer mehr spanne.
        float x = mix(d * 3.4 + 0.5, 3.0, u_emissive) + (b - 0.5) * 0.9;
        t = clamp(int(floor(x)), 0, 3);
    } else {
        t = lit < 0.45 + (b - 0.5) * 0.10 ? 0 : (lit < 0.62 ? 1 : (lit < 0.85 ? 2 : 3));
    }
    vec3 col = ramp(lifted, t);
    if (abs(v_tag - 0.7) < 0.05) {
        col = mix(col, ramp(lifted * 0.8, t), 0.6);   // holm: nur eine spur heller als das paneel
    }
    // Licht der fahne, je ZELLE: die zellmitte im raum der lichter, gegen
    // die normale der zelle. Die lichter liegen leicht ueber der ebene,
    // damit auch die oberseite etwas abbekommt (umgriff 0.35), und fallen
    // mit dem abstand weich ab (radius = groesse der quelle). Nur die
    // farbe aendert sich -- die regions-kennung bleibt, sonst zoegen sich
    // naehte durch den schein.
    vec2 ps = vec2(u_head.x * v_local.x - u_head.y * v_local.y,
                   u_head.y * v_local.x + u_head.x * v_local.y);
    float glow = 0.0;
    for (int i = 0; i < 8; ++i) {
        if (i >= u_plight_n) break;
        vec4 L = u_plights[i];
        vec3 dl = vec3(L.xy - ps, 9.0);
        float d2 = dot(dl, dl);
        float ndl = max(dot(n, dl * inversesqrt(d2)), 0.0) * 0.65 + 0.35;
        float r = max(L.w, 4.0);
        glow += L.z * ndl / (1.0 + d2 / (r * r * 4.0));
    }
    glow = min(glow, 1.3);
    col = min(col + u_glow * 0.60 * glow, vec3(1.0));
    lit = min(1.0, lit + 0.45 * glow);
    // alpha traegt das LICHT; die auspraegungen lesen daraus fuge und halbton.
    o_cell = vec4(col, lit);
    o_id = vec4(base, v_tag);
}
