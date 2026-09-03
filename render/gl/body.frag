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

out vec4 fragColor;

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

    if (r <= u_core_radius_norm) {
        // Echte kugel-schattierung: die normale des sichtbaren halbraums gegen
        // die lichtrichtung. Steht das licht in der bahnebene (u_light.z == 0),
        // ist das die richtige PHASE -- der terminator laeuft dann durch die
        // scheibenmitte, so wie man ihn von oben auf das system auch saehe.
        float lit = mix(u_ambient + (1.0 - u_ambient) * pow(lambert, 0.85),
                        1.0, u_emissive);

        // Mit vektor-zeichnung ist der koerper fast schwarz -- die scheibe ist
        // dann nur noch grund, die linien tragen das bild.
        vec3 ground = mix(u_base_color, u_base_color * 0.30, u_surface_mix);
        color = ground * lit;

        // Rand, auf der lichtseite heller: haelt die silhouette gegen den
        // hintergrund, auch wenn die nachtseite fast schwarz ist.
        float limb = smoothstep(0.78, 1.0, pr);
        color += u_base_color * limb * (0.10 + 0.90 * lambert)
                 * (0.35 + 0.65 * u_surface_mix);
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
