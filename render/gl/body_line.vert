#version 330

// Linien der koerper-zeichnung (gitternetz, konturen, figuren-umrisse, ringe).
//
// Jedes segment kommt als quad aus vier ecken (sechs vertices). Die breite
// wird ERST NACH der skalierung auf den bildschirmradius aufgetragen, also in
// pixeln: eine kontur bleibt beim zoomen gleich dick, waehrend der planet
// waechst. Genau deshalb bleibt die zeichnung vektor und wird nie zur textur.
//
// `a_half` ist die halbe breite plus 0.5 px saum -- dieser saum traegt die
// kantenglaettung im fragment-shader.

in vec2 a_pos;      // einheitskreis, y nach OBEN
in vec3 a_nrm;
in vec3 a_col;
in float a_alpha;
in float a_dark;
in vec2 a_dir;      // normierte segmentrichtung, einheitskreis-raum
in float a_side;    // -1 / +1  quer zur linie
in float a_ext;     // -1 / +1  laengs (quadratische kappe, schliesst gehrungen)
in float a_half;    // halbe breite inkl. saum, in pixeln

uniform vec2 u_center_px;
uniform float u_radius_px;
uniform vec2 u_viewport;
uniform vec3 u_light;
uniform float u_light_exp;
uniform float u_fade;
uniform float u_emissive;
// Kantenlaenge der koerper-pixel in bildschirm-pixeln, 0 = glatte vektor-
// linie. Ab 1 rastert die linie hart auf das grobe raster (siehe
// `_begin_body_layer`): die breite wird mindestens eine halbe zelle, und
// was die linie dabei an breite gewinnt, verliert sie an deckkraft.
uniform float u_cell;

out vec3 v_color;
out float v_alpha;
out float v_dist;   // abstand zur linienmitte, in pixeln
out float v_core;   // halbe breite OHNE saum

void main() {
    vec2 pos_px = u_center_px + vec2(a_pos.x, -a_pos.y) * u_radius_px;

    // Die abbildung einheitskreis -> bildschirm ist eine skalierung mit
    // y-spiegelung. Eine spiegelung ist laengentreu, also bleibt die im
    // einheitskreis normierte richtung auch in pixeln normiert.
    vec2 dir = vec2(a_dir.x, -a_dir.y);
    vec2 nrm = vec2(-dir.y, dir.x);
    float half_w = a_half;
    float ink = 1.0;
    if (u_cell > 0.0) {
        // Harte kante: der rasterizer entscheidet je zellmitte, das quad ist
        // die linie selbst, ohne saum. Eine duenne linie wird so eine kette
        // ganzer zellen statt eines halb deckenden schleiers. Ihre tinte je
        // laenge bleibt erhalten (breite x alpha), mit einer untergrenze,
        // damit eine haarlinie nicht ganz verschwindet.
        float core = max(a_half - 0.5, 0.0);
        half_w = max(core, 0.5 * u_cell);
        ink = clamp(2.0 * core / u_cell, 0.45, 1.0);
    }
    pos_px += nrm * (a_side * half_w) + dir * (a_ext * half_w);

    gl_Position = vec4(
        (pos_px.x / u_viewport.x) * 2.0 - 1.0,
        1.0 - (pos_px.y / u_viewport.y) * 2.0,
        0.0, 1.0
    );

    float d = max(dot(a_nrm, u_light), 0.0);
    float lit = mix(pow(d, u_light_exp), 1.0, u_emissive);
    v_color = a_col;
    v_alpha = a_alpha * mix(a_dark, 1.0, lit) * u_fade * ink;
    v_dist = a_side * half_w;
    // Im raster deckt jedes fragment im quad voll (coverage = 1).
    v_core = (u_cell > 0.0) ? half_w + 0.5 : a_half - 0.5;
}
