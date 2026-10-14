#version 330

// Zusammenbau eines koerpers aus seinem zellraster, in VOLLER aufloesung.
// Hier entsteht, was ein hochgezogenes bild nie haben kann:
//  - jede zelle ist eine LED: gefastes quadrat mit 1-px-spalt (fuge in der
//    dunkelsten farbe des koerpers), und ihre groesse ist der halbton;
//  - 1-px-striche in voller aufloesung: zwischen regionen ungleicher
//    grundfarbe (in der fuge) und am beleuchteten rand;
//  - ausserhalb: eine punktreihe atmosphaere und der halo als punktmatrix.
//
// Koordinaten: top-down-pixel; u_origin ist die linke obere ecke der zelle
// (0, 0) des zellpuffers, eine GANZE pixelposition.

uniform sampler2D u_cells;     // einheit 0: farbe + halbton
uniform sampler2D u_ids;       // einheit 1: regions-kennung + belegt
uniform vec2 u_target;         // groesse des ziels in pixeln
uniform vec2 u_origin;
uniform ivec2 u_grid;          // belegte zellen im puffer
uniform int u_k;               // zellkante in pixeln
uniform vec2 u_center;         // koerpermitte, top-down, ganze pixel
uniform float u_radius;        // kernradius in pixeln
uniform vec3 u_light;          // zur lichtquelle, scheiben-raum (y nach oben)
uniform float u_emissive;
uniform int u_kind;            // 0 planet, 1 stern, 2 gasriese
uniform float u_seed;
uniform vec3 u_grout;          // fugenfarbe
uniform vec3 u_rim;            // strich am beleuchteten rand
uniform vec4 u_atmos;          // farbe, a > 0.5 = hat atmosphaere
uniform vec4 u_halo;           // farbe, deckkraft
uniform float u_halo_radius;   // in kernradien; <= 1 = kein halo
uniform float u_halo_density;

out vec4 fragColor;

float bayer4(ivec2 p) {
    int i = (p.x & 3) + 4 * (p.y & 3);
    int m[16] = int[16](0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5);
    return (float(m[i]) + 0.5) / 16.0;
}

float row_hash(int row, float seed) {
    uint h = uint(row) * 668265263u + uint(seed) * 2246822519u + 97u;
    h = (h ^ (h >> 13u)) * 1274126177u;
    h ^= h >> 16u;
    return float(h & 0xFFFFu) / 65535.0;
}

bool fetch_raw(ivec2 c, out vec4 cell, out vec4 id) {
    if (c.x < 0 || c.y < 0 || c.x >= u_grid.x || c.y >= u_grid.y) {
        cell = vec4(0.0);
        id = vec4(0.0);
        return false;
    }
    ivec2 t = ivec2(c.x, u_grid.y - 1 - c.y);   // textur zaehlt von unten
    cell = texelFetch(u_cells, t, 0);
    id = texelFetch(u_ids, t, 0);
    return id.a > 0.5;
}

// Gasriese: ganze zeilen ruecken um eine zelle zur seite, wie eine
// verschobene bildzeile. Nur im inneren -- die silhouette bleibt rund.
bool fetch(ivec2 c, out vec4 cell, out vec4 id) {
    bool on = fetch_raw(c, cell, id);
    if (!on || u_kind != 2) {
        return on;
    }
    float h = row_hash(c.y - u_grid.y / 2, u_seed);
    int s = h < 0.16 ? -1 : (h > 0.84 ? 1 : 0);
    if (s != 0) {
        vec4 c2, i2;
        if (fetch_raw(c - ivec2(s, 0), c2, i2)) {
            cell = c2;
            id = i2;
        }
    }
    return true;
}

// Ein punkt der groesse n in der mitte der zelle (ink-feld 0..s-1).
bool dot_at(ivec2 sub, int s, int n) {
    int off = (s - n) / 2;
    ivec2 q = sub - ivec2(off);
    return q.x >= 0 && q.y >= 0 && q.x < n && q.y < n;
}

void main() {
    vec2 ptd = vec2(gl_FragCoord.x, u_target.y - gl_FragCoord.y);
    ivec2 rel = ivec2(floor(ptd - u_origin));
    int k = u_k;
    ivec2 c = ivec2(floor(vec2(rel) / float(k)));
    ivec2 sub = rel - c * k;
    int s = k - 1;                         // ink-feld, die letzte zeile/spalte ist die fuge

    vec4 cell, id;
    if (fetch(c, cell, id)) {
        // --- 1. die zelle: gefastes quadrat, halbton = groesse ----------
        int lvl = int(cell.a * 4.0 + 0.5);
        int n = s;
        if (u_kind != 1) {
            if (lvl == 3) n = max(s - 2, 1);
            else if (lvl == 2) n = max(s - 4, 1);
            else if (lvl <= 1) n = 1;
        }
        int off = (s - n) / 2;
        ivec2 q = sub - ivec2(off);
        bool ink = q.x >= 0 && q.y >= 0 && q.x < n && q.y < n;
        if (ink && n >= 3) {
            int m = n - 1;
            int cut = n >= 7 ? 2 : 1;
            if (q.x + q.y < cut || (m - q.x) + q.y < cut
                    || q.x + (m - q.y) < cut || (m - q.x) + (m - q.y) < cut) {
                ink = false;
            }
        }
        vec3 col = ink ? cell.rgb : mix(u_grout, cell.rgb, 0.55);

        // --- 2. striche in voller aufloesung ---------------------------
        vec2 lxy = length(u_light.xy) > 0.05 ? normalize(u_light.xy) : vec2(0.0);
        ivec2 dirs[4] = ivec2[4](ivec2(-1, 0), ivec2(1, 0), ivec2(0, -1), ivec2(0, 1));
        bool edge[4] = bool[4](sub.x == 0, sub.x == k - 1, sub.y == 0, sub.y == k - 1);
        for (int i = 0; i < 4; ++i) {
            if (!edge[i]) {
                continue;
            }
            vec4 nc, ni;
            bool non = fetch(c + dirs[i], nc, ni);
            if (!non) {
                // Silhouette: auf der lichtseite ein heller strich, der den
                // zelltreppen folgt; auf der nachtseite bleibt die fuge.
                vec2 nrm = vec2(float(dirs[i].x), -float(dirs[i].y));
                if (u_emissive < 0.5 && dot(nrm, lxy) > 0.15) {
                    col = u_rim;
                }
            } else if (u_kind != 1 && i % 2 == 1
                       && distance(id.rgb, ni.rgb) > 0.008) {
                // Regionsgrenze: rechts/unten, also IN der fuge -- jede
                // gemeinsame kante genau einmal und ohne luecke.
                // Auf der nachtseite (halbton) traegt der strich nur die
                // hellere farbe, keinen glanz -- sonst laege dort ein netz.
                vec3 hi = max(cell.rgb, nc.rgb);
                float day = clamp(min(cell.a, nc.a) * 4.0 - 3.0, 0.0, 1.0);
                col = mix(mix(u_grout, hi, 0.75), min(hi * 1.35 + 0.025, vec3(1.0)), day);
            }
        }
        fragColor = vec4(col, 1.0);
        return;
    }

    // --- 3. ausserhalb des koerpers ---------------------------------------
    vec2 cc = (vec2(c) + 0.5) * float(k) + u_origin;
    vec2 dv = cc - u_center;
    float dist = length(dv) / max(u_radius, 1.0);
    vec2 dir_up = normalize(vec2(dv.x, -dv.y) + 1e-6);

    if (u_atmos.a > 0.5 && dist < 1.0 + 2.5 * float(k) / max(u_radius, 1.0)) {
        // Eine reihe halber punkte direkt am rand, auf der lichtseite.
        vec4 nc, ni;
        bool touch = false;
        for (int i = 0; i < 4; ++i) {
            ivec2 d = i == 0 ? ivec2(-1, 0) : (i == 1 ? ivec2(1, 0)
                      : (i == 2 ? ivec2(0, -1) : ivec2(0, 1)));
            touch = touch || fetch_raw(c + d, nc, ni);
        }
        float side = u_emissive >= 0.5 ? 1.0 : dot(dir_up, u_light.xy);
        if (touch && side > -0.15 && dot_at(sub, s, max(s / 2, 1))) {
            fragColor = vec4(u_atmos.rgb, 0.85);
            return;
        }
    }

    if (u_halo_radius > 1.0 && dist > 1.0 && dist < u_halo_radius) {
        // Halo als punktmatrix: geordnet ausgeduennt (bayer), zur lichtseite
        // dichter. Ein punkt je zelle, mitte der zelle.
        float t = (dist - 1.0) / (u_halo_radius - 1.0);
        float bias = u_emissive >= 0.5 ? 1.0
                     : clamp(0.55 + 0.45 * dot(dir_up, u_light.xy), 0.0, 1.0);
        float dens = u_halo_density * (1.0 - t) * (1.0 - t) * bias;
        int n = (u_kind == 1 && k >= 5) ? 2 : 1;
        if (bayer4(c) < dens && dot_at(sub, s, n)) {
            fragColor = vec4(u_halo.rgb, u_halo.a * (1.0 - 0.6 * t));
            return;
        }
    }
    discard;
}
