#version 330

// Ortho-Konvention: repliziert das frühere fixed-function
// gluOrtho2D(0, w, 0, h) exakt -- y wächst nach OBEN, KEIN y-flip.
// Gegenstück zu line.vert (top-down, y geflippt). Der unterschied der
// beiden konventionen ist absichtlich und dokumentiert (CLAUDE.md,
// render-convention caveat): schiffspfeil, labels und HUD leben in
// dieser konvention, die orbital-vektoren/linien in der von line.vert.

in vec2 a_pos;
uniform vec2 u_viewport;

void main() {
    vec2 ndc = vec2(
        (a_pos.x / u_viewport.x) * 2.0 - 1.0,
        (a_pos.y / u_viewport.y) * 2.0 - 1.0
    );
    gl_Position = vec4(ndc, 0.0, 1.0);
}
