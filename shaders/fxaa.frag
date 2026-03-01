#version 120

// FXAA 3.11 Fragment Shader (vereinfachte Version)
// Fast Approximate Anti-Aliasing

uniform sampler2D u_texture;
uniform vec2 u_resolution;

varying vec2 v_texcoord;

// Luminanz berechnen
float luminance(vec3 c) {
    return dot(c, vec3(0.299, 0.587, 0.114));
}

void main() {
    vec2 texel_size = 1.0 / u_resolution;
    vec2 uv = v_texcoord;
    
    // Zentraler Pixel
    vec3 center = texture2D(u_texture, uv).rgb;
    float center_luma = luminance(center);
    
    // Nachbar-Pixel (4-Direction Sampling)
    vec3 nw = texture2D(u_texture, uv + vec2(-1.0, -1.0) * texel_size).rgb;
    vec3 ne = texture2D(u_texture, uv + vec2(1.0, -1.0) * texel_size).rgb;
    vec3 sw = texture2D(u_texture, uv + vec2(-1.0, 1.0) * texel_size).rgb;
    vec3 se = texture2D(u_texture, uv + vec2(1.0, 1.0) * texel_size).rgb;
    
    float luma_nw = luminance(nw);
    float luma_ne = luminance(ne);
    float luma_sw = luminance(sw);
    float luma_se = luminance(se);
    
    // Luminanz-Range berechnen
    float luma_min = min(center_luma, min(min(luma_nw, luma_ne), min(luma_sw, luma_se)));
    float luma_max = max(center_luma, max(max(luma_nw, luma_ne), max(luma_sw, luma_se)));
    float luma_range = luma_max - luma_min;
    
    // Wenn der Range zu klein ist, kein Anti-Aliasing nötig
    if (luma_range < 0.0312) {
        gl_FragColor = vec4(center, 1.0);
        return;
    }
    
    // Gradient berechnen
    float gradient_nw_se = abs(luma_nw - luma_se);
    float gradient_ne_sw = abs(luma_ne - luma_sw);
    
    // Lokalen Kontrast bestimmen
    float contrast = max(gradient_nw_se, gradient_ne_sw);
    
    // Wenn Kontrast zu niedrig, kein AA
    if (contrast < 0.0625) {
        gl_FragColor = vec4(center, 1.0);
        return;
    }
    
    // Edge Detection
    vec2 dir;
    dir.x = -((luma_nw + luma_ne) - (luma_sw + luma_se));
    dir.y = ((luma_nw + luma_sw) - (luma_ne + luma_se));
    
    float dir_reduce = max((luma_nw + luma_ne + luma_sw + luma_se) * 0.25, 0.125);
    float rcp_dir_min = 1.0 / (min(abs(dir.x), abs(dir.y)) + dir_reduce);
    
    dir = min(vec2(8.0), max(vec2(-8.0), dir * rcp_dir_min)) * texel_size;
    
    // Blending entlang der Kante
    vec3 result_a = 0.5 * (
        texture2D(u_texture, uv + dir * (1.0/3.0 - 0.5)).rgb +
        texture2D(u_texture, uv + dir * (2.0/3.0 - 0.5)).rgb
    );
    vec3 result_b = result_a * 0.5 + 0.25 * (
        texture2D(u_texture, uv + dir * -0.5).rgb +
        texture2D(u_texture, uv + dir * 0.5).rgb
    );
    
    float luma_b = luminance(result_b);
    
    // Finale Farbe
    if (luma_b < luma_min || luma_b > luma_max) {
        gl_FragColor = vec4(result_a, 1.0);
    } else {
        gl_FragColor = vec4(result_b, 1.0);
    }
}
