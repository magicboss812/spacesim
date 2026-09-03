#version 330

// Vollbild-quad fuer die hintergrund-ebene. Nutzt das geteilte einheits-quad
// (_ensure_quad_vbo), das bereits in NDC liegt -- hier ist also nichts zu
// rechnen. Die ganze arbeit steckt in background.frag.

in vec2 a_pos;

void main() {
    gl_Position = vec4(a_pos, 0.0, 1.0);
}
