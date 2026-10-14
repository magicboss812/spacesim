#version 330

// Zusammenbau eines koerpers aus seinem zellraster, in VOLLER aufloesung.
//  - jede zelle ist eine LED: gefastes quadrat; die 1-px-fuge zeigt sich nur
//    auf der lichtseite und am rand -- innen liest der koerper als flaeche;
//  - der halbton ist die zellgroesse;
//  - 1-px-striche zwischen regionen, die sich in der HELLIGKEIT klar
//    unterscheiden, und am beleuchteten rand;
//  - ausserhalb: atmosphaere als punktreihe, halo als punktmatrix, beim
//    stern davor zwei gestufte glutringe.
//  - u_dissolve: beim wechsel der zellgroesse werden zwei fassungen gezeichnet
//    und je GROSSER zelle (bayer) aufgeteilt -- ein pixel-dissolve statt
//    eines sprungs.
//
// Koordinaten: top-down-pixel; u_origin ist die linke obere ecke der zelle
// (0, 0) des zellpuffers, eine GANZE pixelposition.

uniform sampler2D u_cells;     // einheit 0: farbe + halbton
uniform sampler2D u_ids;       // einheit 1: regions-kennung + kennzeichen
uniform vec2 u_target;
uniform vec2 u_origin;
uniform ivec2 u_grid;
uniform int u_k;
uniform int u_k_big;           // zelle, nach der der dissolve aufteilt
uniform vec3 u_dissolve;       // lo, hi, an (> 0.5): pixel nur, wenn lo <= bayer < hi
uniform vec2 u_center;
uniform float u_radius;
uniform vec3 u_light;
uniform float u_light_exp;
uniform float u_emissive;
uniform int u_kind;            // 0 planet, 1 stern, 2 gasriese
uniform vec3 u_grout;
uniform vec3 u_rim;
uniform vec4 u_atmos;
uniform vec4 u_halo;
uniform float u_halo_radius;
uniform float u_halo_density;

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
    ivec2 t = ivec2(c.x, u_grid.y - 1 - c.y);   // textur zaehlt von unten
    cell = texelFetch(u_cells, t, 0);
    id = texelFetch(u_ids, t, 0);
    return id.a > 0.1;
}

bool dot_at(ivec2 sub, int s, int n) {
    int off = (s - n) / 2;
    ivec2 q = sub - ivec2(off);
    return q.x >= 0 && q.y >= 0 && q.x < n && q.y < n;
}

// Strich an einer kante der zelle c in richtung d. `owner`: nur die rechte
// und untere kante tragen regionsgrenzen, damit jede genau einmal gezogen
// wird (in der fuge, ohne luecke).
vec3 edge_stroke(ivec2 c, ivec2 d, bool owner, vec4 cell, vec4 id, vec3 grout, vec3 col) {
    vec4 nc, ni;
    if (!fetch(c + d, nc, ni)) {
        // Silhouette: auf der lichtseite ein heller strich entlang der
        // zelltreppe; auf der nachtseite bleibt die fuge.
        vec2 lxy = length(u_light.xy) > 0.05 ? normalize(u_light.xy) : vec2(0.0);
        if (u_emissive < 0.5 && dot(vec2(float(d.x), -float(d.y)), lxy) > 0.15) {
            return u_rim;
        }
        return col;
    }
    if (owner && id.a > 0.7 && abs(id.a - ni.a) < 0.05
            && abs(luma(id.rgb) - luma(ni.rgb)) > 0.045) {
        // Nur zwischen gleichen kennzeichen (kein strich entlang eines
        // uebergangs) und nur bei klarem HELLIGKEITS-unterschied.
        vec3 hi = max(cell.rgb, nc.rgb);
        float day = clamp(min(cell.a, nc.a) * 4.0 - 3.0, 0.0, 1.0);
        return mix(mix(grout, hi, 0.75), min(hi * 1.35 + 0.025, vec3(1.0)), day);
    }
    return col;
}

void main() {
    vec2 ptd = vec2(gl_FragCoord.x, u_target.y - gl_FragCoord.y);
    ivec2 rel = ivec2(floor(ptd - u_origin));
    int k = u_k;

    if (u_dissolve.z > 0.5) {
        // Radialer wisch nach der GROSSEN zelle relativ zur koerpermitte:
        // beide fassungen haengen an derselben mitte, und die grosse zelle
        // ist genau 2 x 2 kleine -- die grenze laeuft also auf beiden rastern
        // sauber. Innen die neuere fassung (die grosse beim heranzoomen).
        vec2 bigc = (floor((ptd - u_center) / float(u_k_big)) + 0.5) * float(u_k_big);
        float key = min(length(bigc) / max(u_radius, 1.0), 0.9999);
        if (key < u_dissolve.x || key >= u_dissolve.y) {
            discard;
        }
    }

    ivec2 c = ivec2(floor(vec2(rel) / float(k)));
    ivec2 sub = rel - c * k;
    int s = k - 1;                         // ink-feld, letzte zeile/spalte = fuge

    // Kugelpunkt unter der zellmitte: licht und randnaehe fuer fuge/strich.
    vec2 cc = (vec2(c) + 0.5) * float(k) + u_origin;
    vec2 dv = cc - u_center;
    float dist = length(dv) / max(u_radius, 1.0);
    vec2 pu = vec2(dv.x, -dv.y) / max(u_radius, 1.0);
    vec3 nrm3 = vec3(pu, sqrt(max(0.0, 1.0 - dot(pu, pu))));
    float lit = mix(pow(max(dot(nrm3, u_light), 0.0), u_light_exp), 1.0, u_emissive);

    float cellr = float(k) / max(u_radius, 1.0);
    float reach = max(u_halo_radius, 1.0 + 3.5 * cellr);
    if (dist > reach) {
        discard;                            // schnell raus: weder koerper noch halo
    }

    vec4 cell, id;
    if (fetch(c, cell, id)) {
        // (Waisen sind schon im zellraster weggeputzt: body_cell_clean.frag.)
        // --- 1. die zelle -------------------------------------------------
        int lvl = int(cell.a * 4.0 + 0.5);
        int n = s;
        if (u_kind != 1) {
            if (lvl == 3) n = max(s - 2, 1);
            else if (lvl == 2) n = max(s - 4, 1);
            else if (lvl <= 1) n = 1;
        }
        int off = (s - n) / 2;
        ivec2 q = sub - ivec2(off);
        bool in_field = sub.x < s && sub.y < s;
        bool ink = q.x >= 0 && q.y >= 0 && q.x < n && q.y < n;
        // Die fuge zeigt sich nur auf der lichtseite und zum rand hin; im
        // inneren verschwindet sie in der zellfarbe -- dort liest der koerper
        // als FLAECHE, und das raster wird nicht zur textur ueber allem.
        float gap = max(smoothstep(0.55, 0.85, lit), smoothstep(0.80, 0.97, dist));
        if (u_kind == 1) gap = 0.35 * smoothstep(0.70, 0.98, dist);
        if (ink && n >= 3 && gap > 0.25) {
            int m = n - 1;
            if (q.x + q.y < 1 || (m - q.x) + q.y < 1
                    || q.x + (m - q.y) < 1 || (m - q.x) + (m - q.y) < 1) {
                ink = false;
            }
        }
        vec3 grout = mix(u_grout, cell.rgb, 0.45);
        vec3 col;
        if (ink) {
            col = cell.rgb;
        } else if (in_field) {
            col = grout;                    // halbton: um den punkt die fuge
        } else {
            col = mix(cell.rgb, grout, gap);
        }

        // --- 2. striche ---------------------------------------------------
        // Nur randpixel einer zelle fragen ihren nachbarn; ein pixel liegt an
        // hoechstens zwei kanten. (Ohne lokale arrays und schleife: im
        // software-GL kosteten die den groessten teil des passes.)
        if (sub.x == 0) col = edge_stroke(c, ivec2(-1, 0), false, cell, id, grout, col);
        else if (sub.x == k - 1) col = edge_stroke(c, ivec2(1, 0), true, cell, id, grout, col);
        if (sub.y == 0) col = edge_stroke(c, ivec2(0, -1), false, cell, id, grout, col);
        else if (sub.y == k - 1) col = edge_stroke(c, ivec2(0, 1), true, cell, id, grout, col);
        fragColor = vec4(col, 1.0);
        return;
    }

    // --- 3. ausserhalb des koerpers ---------------------------------------
    vec2 dir_up = normalize(pu + 1e-6);

    if (u_kind == 1 && dist > 1.0) {
        // Gestufte glut direkt am sternenrand: zwei ringe aus ganzen zellen,
        // erst gross und hell, dann kleiner -- danach die punktmatrix.
        if (dist < 1.0 + 1.6 * cellr && dot_at(sub, s, max(s - 1, 1))) {
            fragColor = vec4(mix(u_halo.rgb, vec3(1.0), 0.25), 0.92);
            return;
        }
        if (dist < 1.0 + 3.2 * cellr && dot_at(sub, s, max(s - 3, 1))) {
            fragColor = vec4(u_halo.rgb, 0.80);
            return;
        }
    }

    if (u_atmos.a > 0.5 && dist < 1.0 + 1.6 * cellr) {
        vec4 nc, ni;
        bool touch = fetch(c + ivec2(-1, 0), nc, ni) || fetch(c + ivec2(1, 0), nc, ni)
                  || fetch(c + ivec2(0, -1), nc, ni) || fetch(c + ivec2(0, 1), nc, ni);
        float side = u_emissive >= 0.5 ? 1.0 : dot(dir_up, u_light.xy);
        if (touch && side > -0.15 && dot_at(sub, s, max(s / 2, 1))) {
            fragColor = vec4(u_atmos.rgb, 0.85);
            return;
        }
    }

    if (u_halo_radius > 1.0 && dist > 1.0 && dist < u_halo_radius) {
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
