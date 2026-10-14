#version 330

in vec3 v_color;
in float v_alpha;
in float v_dist;
in float v_core;

uniform vec2 u_center_px;
uniform vec2 u_viewport;
uniform float u_cell;
uniform float u_seed;
uniform float u_jitter;

out vec4 fragColor;

float cell_hash(ivec2 p, float seed) {
    uint h = uint(p.x) * 374761393u + uint(p.y) * 668265263u
             + uint(seed) * 2246822519u;
    h = (h ^ (h >> 13u)) * 1274126177u;
    h ^= h >> 16u;
    return float(h & 0xFFFFu) / 65535.0;
}

void main() {
    // Exakt ein pixel kantenglaettung: die deckung faellt ueber den 0.5-px-saum
    // von 1 auf 0. Ohne das flimmern die 0.6-px-gitterlinien beim zoomen.
    float coverage = clamp(v_core + 0.5 - abs(v_dist), 0.0, 1.0);
    float alpha = v_alpha * coverage;
    if (alpha <= 0.002) {
        discard;
    }
    vec3 color = v_color;
    if (u_cell > 0.0) {
        // Dieselbe zellstreuung wie die fuellungen (body_surface.frag),
        // nur mit anderem salz: eine kontur soll nicht exakt die helligkeit
        // der zelle darunter nachzeichnen.
        vec2 ref = vec2(u_center_px.x, u_viewport.y - u_center_px.y) / u_cell;
        ivec2 c = ivec2(floor(gl_FragCoord.xy)) - ivec2(floor(ref + 0.5));
        color *= 1.0 + u_jitter * (cell_hash(c, u_seed + 7.0) * 2.0 - 1.0);
    }
    fragColor = vec4(color, alpha);
}
