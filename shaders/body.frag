#version 120

varying vec2 v_local;

uniform vec3 u_base_color;
uniform vec3 u_atmos_color;
uniform float u_core_radius_norm;
uniform float u_atmos_radius_norm;
uniform float u_atmos_alpha;
uniform float u_glow_alpha;

void main() {
    float r = length(v_local);
    if (r > 1.0) {
        discard;
    }

    vec3 color = vec3(0.0, 0.0, 0.0);
    float alpha = 0.0;

    if (r <= u_core_radius_norm) {
        color = u_base_color;
        alpha = 1.0;
    } else {
        if (u_atmos_alpha > 0.0 && u_atmos_radius_norm > u_core_radius_norm && r <= u_atmos_radius_norm) {
            float t_atmos = (u_atmos_radius_norm - r) / max(0.0001, (u_atmos_radius_norm - u_core_radius_norm));
            float a_atmos = u_atmos_alpha * clamp(t_atmos, 0.0, 1.0);
            color += u_atmos_color * a_atmos;
            alpha += a_atmos;
        }

        if (u_glow_alpha > 0.0 && r >= u_core_radius_norm) {
            float t_glow = (1.0 - r) / max(0.0001, (1.0 - u_core_radius_norm));
            float a_glow = u_glow_alpha * clamp(t_glow, 0.0, 1.0);
            color += u_base_color * a_glow;
            alpha += a_glow;
        }

        alpha = clamp(alpha, 0.0, 1.0);
        if (alpha <= 0.0001) {
            discard;
        }
        color /= alpha;
    }

    gl_FragColor = vec4(color, alpha);
}
