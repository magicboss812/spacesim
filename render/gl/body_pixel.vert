#version 330

// Zusammenbau eines koerpers: ein quad ueber sein rechteck (u_rect, NDC:
// xmin, ymin, xmax, ymax), kern plus halo, auf den schirm geklemmt.
in vec2 a_pos;
uniform vec4 u_rect;

void main() {
    vec2 t = a_pos * 0.5 + 0.5;
    gl_Position = vec4(mix(u_rect.xy, u_rect.zw, t), 0.0, 1.0);
}
