#version 120

// Vertex Shader für Body-Rendering (Kreis, Atmosphäre, Glow)
// Rendert ein Quad, das Fragment-Shader macht den Rest

attribute vec2 a_position;
attribute vec2 a_texcoord;

varying vec2 v_texcoord;
varying vec2 v_position;

void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
    v_texcoord = a_texcoord;
    v_position = a_position;
}
