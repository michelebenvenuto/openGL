import pygame
import numpy

from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

pygame.init()
screen = pygame.display.set_mode((800,600), pygame.OPENGL | pygame.DOUBLEBUF)



vertex_shader = """
#version 460

layout (location = 0) in vec3 position;

void main()
{
    gl_Position = vec4(position.x, position.y, position.z, 1.0f);
}
"""

fragment_shader = """
#version 460

layout (location = 0) out vec4 fragColor;

void main()
{
    fragColor = vec4(1.0f, 0.5f, 0.5f, 1.0f);
}
"""
shader = compileProgram(
    compileShader(vertex_shader, GL_VERTEX_SHADER),
    compileShader(fragment_shader, GL_FRAGMENT_SHADER)
)

vertex_data = numpy.array([
    -0.5, -0.5, 0.0,
     0.5, -0.5, 0.0,
     0.0,  0.5, 0.0
], dtype=numpy.float32)

vertex_buffer_object = glGenBuffers(1)

glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_object)
glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)

vertex_array_object = glGenVertexArrays(1)
glBindVertexArray(vertex_array_object)

glVertexAttribPointer(
    0, #location
    3, #size
    GL_FLOAT, #type
    GL_FALSE, #normalized?
    3 * 4,
    ctypes.c_void_p(0)
)
glEnableVertexAttribArray(0)

running = True
while running:
    glClearColor(0.5, 1.0, 0.5, 1.0)
    
    glUseProgram(shader)
    glDrawArrays(GL_TRIANGLES, 0, 3)
    

    pygame.display.flip()


    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False