"""
Shader Manager für OpenGL-Shader-Kompilierung und -Verwaltung.
"""

from OpenGL.GL import (
    glCreateShader, glShaderSource, glCompileShader,
    glCreateProgram, glAttachShader, glLinkProgram,
    glDeleteShader, glGetShaderInfoLog, glGetProgramInfoLog,
    glGetShaderiv, glGetProgramiv, glUseProgram,
    GL_VERTEX_SHADER, GL_FRAGMENT_SHADER, GL_COMPILE_STATUS, GL_LINK_STATUS
)
from OpenGL.GL import glGetUniformLocation, glUniform1f, glUniform2f, glUniform3f, glUniform4f


class ShaderManager:
    """Verwaltet OpenGL-Shader: Kompilierung, Linking und Uniform-Übergabe."""
    
    def __init__(self):
        self.programs = {}
    
    def compile_shader(self, source: str, shader_type) -> int:
        """
        Kompiliert einen einzelnen Shader.
        
        Args:
            source: GLSL-Quellcode als String
            shader_type: GL_VERTEX_SHADER oder GL_FRAGMENT_SHADER
            
        Returns:
            Shader-ID
        """
        shader = glCreateShader(shader_type)
        glShaderSource(shader, source)
        glCompileShader(shader)
        
        # Fehlerprüfung
        if not glGetShaderiv(shader, GL_COMPILE_STATUS):
            error = glGetShaderInfoLog(shader).decode('utf-8')
            raise RuntimeError(f"Shader compilation error:\n{error}")
        
        return shader
    
    def create_program(self, name: str, vertex_source: str, fragment_source: str) -> int:
        """
        Erstellt ein Shader-Programm aus Vertex- und Fragment-Shader.
        """
        vertex_shader = self.compile_shader(vertex_source, GL_VERTEX_SHADER)
        fragment_shader = self.compile_shader(fragment_source, GL_FRAGMENT_SHADER)
        
        program = glCreateProgram()
        glAttachShader(program, vertex_shader)
        glAttachShader(program, fragment_shader)
        glLinkProgram(program)
        
        # Fehlerprüfung
        if not glGetProgramiv(program, GL_LINK_STATUS):
            error = glGetProgramInfoLog(program).decode('utf-8')
            raise RuntimeError(f"Program linking error:\n{error}")
        
        # Shader aufräumen (nicht mehr nötig nach Linking)
        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)
        
        self.programs[name] = program
        return program
    
    def get_program(self, name: str) -> int:
        """Gibt ein gespeichertes Programm zurück."""
        return self.programs.get(name)
    
    def use_program(self, name: str):
        """Aktiviert ein Shader-Programm."""
        program = self.get_program(name)
        if program:
            glUseProgram(program)
    
    @staticmethod
    def set_uniform_1f(program: int, name: str, value: float):
        """Setzt einen float Uniform."""
        location = glGetUniformLocation(program, name)
        glUniform1f(location, value)
    
    @staticmethod
    def set_uniform_2f(program: int, name: str, v1: float, v2: float):
        """Setzt einen vec2 Uniform."""
        location = glGetUniformLocation(program, name)
        glUniform2f(location, v1, v2)
    
    @staticmethod
    def set_uniform_3f(program: int, name: str, v1: float, v2: float, v3: float):
        """Setzt einen vec3 Uniform."""
        location = glGetUniformLocation(program, name)
        glUniform3f(location, v1, v2, v3)
    
    @staticmethod
    def set_uniform_4f(program: int, name: str, v1: float, v2: float, v3: float, v4: float):
        """Setzt einen vec4 Uniform."""
        location = glGetUniformLocation(program, name)
        glUniform4f(location, v1, v2, v3, v4)
