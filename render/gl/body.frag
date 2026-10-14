#version 330

// Grund-scheibe eines koerpers: kugelschattierung, rand, atmosphaere, glow.
// Die prozedurale vektor-zeichnung (body_surface / body_line) liegt DARUEBER
// und traegt ihre eigene, facetten-genaue beleuchtung; hier steht nur, was
// auch ein drei pixel grosser koerper noch braucht.

in vec2 v_local;

uniform vec3 u_base_color;
uniform vec3 u_atmos_color;
uniform float u_core_radius_norm;
uniform float u_atmos_radius_norm;
uniform float u_atmos_alpha;
uniform float u_glow_alpha;

// Richtung ZUR lichtquelle im scheiben-raum (y nach oben). Laenge 1, ausser
// bei selbstleuchtenden koerpern -- dort uebernimmt u_emissive.
uniform vec3 u_light;
uniform float u_ambient;
uniform float u_emissive;
// 0 = flache scheibe in koerperfarbe (alte optik, kleine koerper),
// 1 = dunkler grund, auf dem die vektor-zeichnung sitzt.
uniform float u_surface_mix;

// Koerper-raster (`_begin_body_layer`). u_cell = kantenlaenge einer zelle in
// bildschirm-pixeln, 0 = glatt wie frueher. Jedes fragment ist dann EINE
// zelle; alles unten wird pro zelle relativ zur koerpermitte gerechnet,
// damit beim schwenken nichts ueber den koerper wandert.
uniform vec2 u_center_px;
uniform vec2 u_viewport;
uniform float u_outer_radius_px;
uniform float u_cell;
uniform float u_seed;
uniform float u_jitter;
uniform float u_levels;
// 0 = grundscheibe, 1 = schattierungs-pass UEBER der vektor-zeichnung:
// lichtkante, geditherter nachtschatten, glanzpunkte.
uniform int u_mode;
uniform float u_rim;
uniform float u_shade;

out vec4 fragColor;

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

ivec2 local_cell() {
    vec2 ref = vec2(u_center_px.x, u_viewport.y - u_center_px.y) / u_cell;
    return ivec2(floor(gl_FragCoord.xy)) - ivec2(floor(ref + 0.5));
}

// Schattierungs-pass: nur zellen der kernscheibe, alles andere verworfen.
// Liegt ueber der vektor-zeichnung und gibt ihr, was ein pixel-artist von
// hand setzt -- die zeichnung selbst bleibt unangetastet.
void shade_pass(float pr, float lambert) {
    if (u_cell <= 0.0 || pr >= 1.0) {
        discard;
    }
    ivec2 c = local_cell();
    float b = bayer4(c);
    float core_px = u_outer_radius_px * u_core_radius_norm;
    // Eine zelle breit, gemessen von innen an die kante.
    float rim_band = 1.0 - 1.05 * u_cell / max(core_px, 1e-3);
    vec3 hi = mix(u_base_color, vec3(1.0), 0.55);

    if (u_emissive >= 0.5) {
        // Stern: kein schatten. Ein heller rand-funke hier und da, sonst
        // traegt die zeichnung (randverdunklung, granulation) das bild.
        if (pr > rim_band && cell_hash(c, u_seed + 31.0) < 0.22) {
            fragColor = vec4(hi, 0.35 * u_rim);
            return;
        }
        discard;
    }

    // 1. Lichtkante: die aeusserste zellreihe auf der beleuchteten seite.
    if (pr > rim_band && lambert > 0.18) {
        fragColor = vec4(hi, u_rim * smoothstep(0.18, 0.6, lambert));
        return;
    }
    // 2. Nachtschatten: ein bayer-band NUR am terminator. Zur nacht hin
    // dichter, im tiefen schatten wieder aus -- dort ist die zeichnung schon
    // dunkel, und ein muster ueber der ganzen nachtseite las sich als
    // schachbrett statt als schattierung.
    float band = smoothstep(0.0, 0.06, lambert) * (1.0 - smoothstep(0.08, 0.26, lambert));
    if (b < band * 0.55) {
        fragColor = vec4(u_base_color * 0.08, u_shade);
        return;
    }
    // 3. Glanzpunkte: wenige helle zellen um den subsolaren punkt.
    float gloss = smoothstep(0.82, 1.0, lambert);
    if (gloss > 0.0 && cell_hash(c, u_seed + 13.0) < gloss * 0.30) {
        fragColor = vec4(hi, 0.45 * u_rim * gloss);
        return;
    }
    discard;
}

void main() {
    float r = length(v_local);
    if (r > 1.0) {
        discard;
    }

    vec3 color = vec3(0.0);
    float alpha = 0.0;

    // Scheiben-koordinaten: v_local zaehlt y nach unten (bildschirm),
    // die kugel rechnet y nach oben.
    vec2 p = vec2(v_local.x, -v_local.y) / max(u_core_radius_norm, 1e-6);
    float pr = min(length(p), 1.0);
    float lambert = max(dot(vec3(p, sqrt(max(0.0, 1.0 - pr * pr))), u_light), 0.0);

    if (u_mode == 1) {
        shade_pass(r / max(u_core_radius_norm, 1e-6), lambert);
        return;
    }

    if (r <= u_core_radius_norm) {
        // Echte kugel-schattierung: die normale des sichtbaren halbraums gegen
        // die lichtrichtung. Steht das licht in der bahnebene (u_light.z == 0),
        // ist das die richtige PHASE -- der terminator laeuft dann durch die
        // scheibenmitte, so wie man ihn von oben auf das system auch saehe.
        float lit = mix(u_ambient + (1.0 - u_ambient) * pow(lambert, 0.85),
                        1.0, u_emissive);
        if (u_cell > 0.0 && u_surface_mix < 0.5) {
            // Feste helligkeitsstufen mit bayer-uebergang statt verlauf --
            // nur solange die scheibe das bild TRAEGT. Unter der halb
            // durchsichtigen vektor-zeichnung schien das muster als
            // schachbrett durch die ganze flaeche.
            float levels = max(u_levels, 2.0);
            lit = floor(lit * levels + bayer4(local_cell())) / levels;
        }

        // Mit vektor-zeichnung ist der koerper fast schwarz -- die scheibe ist
        // dann nur noch grund, die linien tragen das bild.
        vec3 ground = mix(u_base_color, u_base_color * 0.30, u_surface_mix);
        color = ground * lit;

        // Rand, auf der lichtseite heller: haelt die silhouette gegen den
        // hintergrund, auch wenn die nachtseite fast schwarz ist.
        float limb = smoothstep(0.78, 1.0, pr);
        color += u_base_color * limb * (0.10 + 0.90 * lambert)
                 * (0.35 + 0.65 * u_surface_mix);
        if (u_cell > 0.0) {
            color *= 1.0 + u_jitter * (cell_hash(local_cell(), u_seed) * 2.0 - 1.0);
        }
        alpha = 1.0;
    } else {
        if (u_atmos_alpha > 0.0 && u_atmos_radius_norm > u_core_radius_norm && r <= u_atmos_radius_norm) {
            float t_atmos = (u_atmos_radius_norm - r) / max(0.0001, (u_atmos_radius_norm - u_core_radius_norm));
            // Die atmosphaere leuchtet nur, wo sonne drauf steht.
            float shade = mix(u_ambient + (1.0 - u_ambient) * lambert, 1.0, u_emissive);
            float a_atmos = u_atmos_alpha * clamp(t_atmos, 0.0, 1.0) * shade;
            color += u_atmos_color * a_atmos;
            alpha += a_atmos;
        }

        if (u_glow_alpha > 0.0) {
            float t_glow = (1.0 - r) / max(0.0001, (1.0 - u_core_radius_norm));
            // Zur lichtseite hin etwas kraeftiger -- derselbe versatz, den das
            // mockup seinem halo gibt.
            float bias = mix(0.80 + 0.20 * dot(normalize(p + 1e-6), u_light.xy),
                             1.0, u_emissive);
            float a_glow = u_glow_alpha * clamp(t_glow, 0.0, 1.0) * clamp(bias, 0.0, 1.0);
            if (u_cell > 0.0) {
                // Der halo in stufen: ringe aus ganzen zellen mit
                // geditherter grenze statt eines weichen verlaufs.
                float steps = max(u_levels, 2.0) * 1.5;
                float g = a_glow / max(u_glow_alpha, 1e-4);
                a_glow = u_glow_alpha * floor(g * steps + bayer4(local_cell())) / steps;
            }
            color += u_base_color * a_glow;
            alpha += a_glow;
        }

        alpha = clamp(alpha, 0.0, 1.0);
        if (alpha <= 0.0001) {
            discard;
        }
        color /= alpha;
    }

    fragColor = vec4(color, alpha);
}
