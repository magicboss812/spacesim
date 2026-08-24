#version 330

// Flaechen der koerper-zeichnung (stufen-fuellungen, figuren-fuellungen).
// Die geometrie liegt im EINHEITSKREIS und wird hier auf den bildschirm-
// radius skaliert -- deshalb bleibt ein koerper bei jeder zoomstufe scharf,
// ohne dass ein vertex neu gerechnet wird.

in vec2 a_pos;      // einheitskreis, y nach OBEN
in vec3 a_nrm;      // 3D-normale der facette (flat: alle drei ecken gleich)
in vec3 a_col;
in float a_alpha;   // deckkraft bei vollem licht
in float a_dark;    // anteil davon, der auf der nachtseite bleibt

uniform vec2 u_center_px;
uniform float u_radius_px;
uniform vec2 u_viewport;
uniform vec3 u_light;      // richtung ZUR lichtquelle, scheiben-raum (y nach oben)
uniform float u_light_exp;
uniform float u_fade;      // einblendung ueber der detailschwelle
uniform float u_emissive;  // 1 = selbstleuchtend (stern)

out vec3 v_color;
out float v_alpha;

void main() {
    // Der bildschirm zaehlt y nach unten, die scheibe nach oben.
    vec2 pos_px = u_center_px + vec2(a_pos.x, -a_pos.y) * u_radius_px;
    gl_Position = vec4(
        (pos_px.x / u_viewport.x) * 2.0 - 1.0,
        1.0 - (pos_px.y / u_viewport.y) * 2.0,
        0.0, 1.0
    );

    float d = max(dot(a_nrm, u_light), 0.0);
    float lit = mix(pow(d, u_light_exp), 1.0, u_emissive);
    v_color = a_col;
    v_alpha = a_alpha * mix(a_dark, 1.0, lit) * u_fade;
}
