#version 330

// Vertex-stufe des UI-SDF-shaders. Ortho-konvention (y nach OBEN, ursprung
// unten links) -- identisch zu ortho.vert und texquad.vert. Die umrechnung
// aus der top-down-konvention der UI-schicht passiert in ui/draw.py, nicht
// hier (siehe CLAUDE.md, render-convention caveat).
//
// INSTANZIERT: alle form-parameter kommen als per-instanz-attribute statt
// als uniforms. Ein HUD-frame besteht aus gut 160 zeichenaufrufen (allein
// der attitude-ring bringt 48 teilstriche mit); als einzel-draws mit je 14
// uniform-schreibvorgaengen war das der groesste einzelposten der UI-zeit.
// Aufeinanderfolgende rechtecke werden jetzt in ui/draw.py gesammelt und in
// EINEM instanzierten draw gezeichnet -- die instanz-reihenfolge entspricht
// der aufruf-reihenfolge, das blending bleibt also exakt gleich.
//
// Das quad wird um i_expand pixel VERGROESSERT gezeichnet, damit schlagschatten
// und die 1px-kantenglaettung platz haben. v_half bleibt dabei die ECHTE halbe
// groesse der form -- der SDF im fragment-shader rechnet also weiter mit dem
// unveraenderten rechteck.

in vec2 a_corner;          // einheits-quad-ecken in [-1, 1] (TRIANGLE_STRIP)

in vec4 i_rect;            // x, y, w, h in ortho-pixeln, (x,y) = untere linke ecke
in float i_expand;         // zusaetzlicher rand in pixeln (schatten + AA)
in float i_rotation;       // rotation um die mitte, radiant, gegen den uhrzeigersinn
in vec4 i_radius;
in vec4 i_fill;
in vec4 i_fill2;
in float i_gradient;
in vec4 i_border_color;
in float i_border_width;
in vec4 i_shadow_color;
in vec2 i_shadow_offset;
in float i_shadow_softness;
in vec2 i_arc;

uniform vec2 u_viewport;

out vec2 v_local;          // position relativ zur mitte, IM rechteck-eigenen system
out vec2 v_half;           // halbe rechteck-groesse in pixeln
flat out vec4 v_radius;
flat out vec4 v_fill;
flat out vec4 v_fill2;
flat out float v_gradient;
flat out vec4 v_border_color;
flat out float v_border_width;
flat out vec4 v_shadow_color;
flat out vec2 v_shadow_offset;
flat out float v_shadow_softness;
flat out vec2 v_arc;

void main() {
    vec2 half_size = i_rect.zw * 0.5;
    vec2 center = i_rect.xy + half_size;
    vec2 local = a_corner * (half_size + i_expand);

    // v_local ist bewusst UNROTIERT: der SDF arbeitet im lokalen system der
    // form, die rotation betrifft nur, wo das quad auf dem bildschirm landet.
    v_local = local;
    v_half = half_size;

    v_radius = i_radius;
    v_fill = i_fill;
    v_fill2 = i_fill2;
    v_gradient = i_gradient;
    v_border_color = i_border_color;
    v_border_width = i_border_width;
    v_shadow_color = i_shadow_color;
    v_shadow_offset = i_shadow_offset;
    v_shadow_softness = i_shadow_softness;
    v_arc = i_arc;

    float c = cos(i_rotation);
    float s = sin(i_rotation);
    vec2 rotated = vec2(local.x * c - local.y * s, local.x * s + local.y * c);

    vec2 pos_px = center + rotated;
    vec2 ndc = (pos_px / u_viewport) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
}
