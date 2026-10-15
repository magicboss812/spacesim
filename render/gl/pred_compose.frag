#version 330

// Zusammenbau der vorhersagelinie: der puffer (`pred_seg.frag`,
// `pred_mark.frag`) als vier schichten, von unten: dunkler saum, schiene,
// linie, marke. Vormultipliziert ausgegeben (blend ONE, 1 - SRC_ALPHA).

uniform sampler2D u_buf;
uniform vec3 u_line;
uniform vec3 u_rail;
uniform vec3 u_mark;
uniform vec3 u_ground;

out vec4 fragColor;

void over(inout vec4 acc, vec3 c, float a) {
    acc.rgb = c * a + acc.rgb * (1.0 - a);
    acc.a = a + acc.a * (1.0 - a);
}

void main() {
    vec4 v = texelFetch(u_buf, ivec2(gl_FragCoord.xy), 0);
    if (v.r + v.g + v.b + v.a <= 0.0) discard;
    vec4 acc = vec4(0.0);
    over(acc, u_ground, v.g);
    over(acc, u_rail, v.b);
    over(acc, u_line, v.r);
    over(acc, u_mark, v.a);
    fragColor = acc;
}
