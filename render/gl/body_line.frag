#version 330

in vec3 v_color;
in float v_alpha;
in float v_dist;
in float v_core;

out vec4 fragColor;

void main() {
    // Exakt ein pixel kantenglaettung: die deckung faellt ueber den 0.5-px-saum
    // von 1 auf 0. Ohne das flimmern die 0.6-px-gitterlinien beim zoomen.
    float coverage = clamp(v_core + 0.5 - abs(v_dist), 0.0, 1.0);
    float alpha = v_alpha * coverage;
    if (alpha <= 0.002) {
        discard;
    }
    fragColor = vec4(v_color, alpha);
}
