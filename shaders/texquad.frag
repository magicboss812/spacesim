#version 330

// Texturierte quads (labels, HUD, UI-text) in der ortho-konvention.
//
// u_color ist eine multiplikative TOENUNG. Der text wird weiss gerastert
// (font.render(..., (255,255,255))) und hier eingefaerbt -- ohne diesen
// uniform waere farbiger text unmoeglich, weil die farbe sonst in der
// pygame-rasterung festgebrannt und damit teil des cache-schluessels waere.
// So genuegt EINE weisse textur pro text, beliebig oft verschieden getoent.
//
// ACHTUNG: uniforms starten in OpenGL bei 0 -- ein aufrufer, der u_color
// nicht setzt, zeichnet unsichtbar. Einziger aufrufer ist
// Renderer._draw_texture_ortho bzw. ui/text.py, beide setzen ihn immer.

uniform sampler2D u_texture;
uniform vec4 u_color;

in vec2 v_uv;
out vec4 fragColor;

void main() {
    fragColor = texture(u_texture, v_uv) * u_color;
}
