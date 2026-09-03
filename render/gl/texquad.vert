#version 330

// Texturierte quads (labels, HUD) in der ortho-konvention (y nach oben),
// wie die früheren immediate-mode glTexCoord/glVertex-quads unter
// gluOrtho2D(0, w, 0, h). u_rect = (x, y, breite, höhe) in pixeln,
// (x, y) = untere linke ecke; texcoord (0,0) liegt ebendort -- passend
// zu pygame.image.tostring(surface, 'RGBA', True) (vertikal geflippt).

in vec2 a_corner;      // einheits-quad-ecken in [-1, 1] (TRIANGLE_STRIP)
uniform vec4 u_rect;   // x, y, w, h in ortho-pixeln
uniform vec2 u_viewport;

out vec2 v_uv;

void main() {
    vec2 t = a_corner * 0.5 + 0.5;
    v_uv = t;
    vec2 pos_px = u_rect.xy + t * u_rect.zw;
    vec2 ndc = (pos_px / u_viewport) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
}
