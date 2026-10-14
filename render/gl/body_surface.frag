#version 330

// Im koerper-raster (u_cell > 0) ist jedes fragment eine ZELLE. Jede zelle
// streut dann leicht in der helligkeit, gesaet je koerper und an die zelle
// relativ zur koerpermitte gebunden -- beim schwenken wandert nichts.
// Die deckkraft wird NICHT gedithert: eine facette ist ohnehin flach
// beleuchtet, und ein bayer-muster ueber jeder fuellung lag als
// schachbrett ueber dem ganzen koerper.

in vec3 v_color;
in float v_alpha;

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

ivec2 local_cell() {
    vec2 ref = vec2(u_center_px.x, u_viewport.y - u_center_px.y) / u_cell;
    return ivec2(floor(gl_FragCoord.xy)) - ivec2(floor(ref + 0.5));
}

void main() {
    float alpha = v_alpha;
    vec3 color = v_color;
    if (u_cell > 0.0) {
        ivec2 c = local_cell();
        color *= 1.0 + u_jitter * (cell_hash(c, u_seed) * 2.0 - 1.0);
    }
    if (alpha <= 0.002) {
        discard;
    }
    fragColor = vec4(color, alpha);
}
