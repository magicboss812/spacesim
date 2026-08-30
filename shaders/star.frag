#version 330

// Sterne sind harte marken auf dem virtuellen pixelraster -- keine
// kantenglaettung. Dieselbe kantenqualitaet wie die anzeigeschrift des HUDs
// (ui/theme.py: "gerastert OHNE kantenglaettung").
//
// `u_pixel_round` formt die einzelne zelle genau wie in background.frag:
// 0 = volles quadrat, 1 = runder leuchtpunkt mit spalt. Ein sprite ist
// v_cells x v_cells zellen gross, deshalb wird `v_uv` damit hochskaliert,
// bevor die zellform greift -- sonst waere ein 2x2-stern EIN grosser punkt
// statt vier kleiner.
//
// `v_uv` kommt aus dem vertex-shader und ist NICHT `gl_PointCoord`: das
// liefert auf dem NVIDIA-treiber dieses rechners konstant (0, 0), womit die
// maske jedes fragment verwarf und das feld bei der Vorgabe
// `pixel_round = 1.0` komplett unsichtbar war. Siehe star.vert.
//
// Die farbe ist die HUD-textfarbe (#dceaf4), nicht das warme weiss des
// urspruenglichen entwurfs: ein warmer stern auf marineblauem grund faellt
// neben cyan-daten sofort als fremdkoerper auf.

uniform float u_pixel_round;

in float v_alpha;
in vec2 v_uv;
in float v_cells;
out vec4 fragColor;

void main() {
    float mask = 1.0;
    if (u_pixel_round > 0.0) {
        vec2 q = abs(fract(v_uv * v_cells) - 0.5) * 2.0;
        float d = mix(max(q.x, q.y), length(q), u_pixel_round);
        mask = step(d, 1.0 - 0.18 * u_pixel_round);
    }
    if (mask <= 0.0) {
        discard;
    }
    // Fuellgrad-ausgleich wie in background.frag: ein gerundeter stern
    // verliert rund 47 % seiner pixel und saehe sonst blasser aus, sobald man
    // nur an u_pixel_round dreht.
    fragColor = vec4(0.862745, 0.917647, 0.956863,
                     min(1.0, v_alpha / mix(1.0, 0.53, u_pixel_round)));
}
