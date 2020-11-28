import pygame
import numpy
import glm
import pyassimp

from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from math import sin


pygame.init()
screen = pygame.display.set_mode((800,600), pygame.OPENGL | pygame.DOUBLEBUF)
clock = pygame.time.Clock()


vertex_shader = """
#version 460

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 texcoords;


uniform mat4 theMatrix;
uniform vec3 light;
uniform vec4 color;

out float intensity;
out vec4 vertexColor;
out vec2 vertexTexcoords;
out vec3 lPosition;

void main()
{
    vertexTexcoords = texcoords;
    intensity = dot(normal, normalize(light));
    gl_Position = theMatrix * vec4(position.x, position.y, position.z, 1.0f);
    vertexColor = color* intensity;
    lPosition = position;
}
"""

fragment_shader = """
#version 460

layout (location = 0) out vec4 fragColor;

in float intensity;
in vec4 vertexColor;
in vec2 vertexTexcoords;

uniform vec4 ambient;
uniform vec4 color;
uniform sampler2D tex;

void main()
{

    fragColor = color * texture(tex, vertexTexcoords)* vertexColor * intensity;
}
"""
rotation_shader = """
#version 460

layout (location = 0) out vec4 fragColor;

in float intensity;
in vec4 vertexColor;
in vec2 vertexTexcoords;

uniform vec4 color;
uniform float time1;
uniform vec4 ambient;

uniform sampler2D tex;

void main()
{
    float angle = time1;
    vec3 color1 = vec3(cos(angle), sin(angle + 3.1415) , sin(angle));
    fragColor =  ambient + color * vec4(color1,1) * texture(tex, vertexTexcoords) * vertexColor * intensity;
}
""" 
position_shader = """
#version 460

layout (location = 0) out vec4 fragColor;

in float intensity;
in vec4 vertexColor;
in vec2 vertexTexcoords;
in vec3 lPosition;

uniform vec4 ambient;
uniform vec4 color;
uniform sampler2D tex;

void main()
{  
    vec4 color1 = lPosition.x > 0? vec4(1.0,0.0,0.0,1.0): vec4(0.0, 1.0, 0.0, 1.0);
    vec4 color2 = lPosition.y > 75? vec4(0.0,0.0,1.0,1.0): vec4(1.0, 0.0, 0, 1.0);
    vec4 color3 = cos(lPosition.z) > 0? vec4(0.0,1.0,0,1.0): vec4(0.0,1.0,1.0,1.0);
    vec4 finalColor = color1 + color2 + color3;
    fragColor = ambient + color * finalColor * texture(tex, vertexTexcoords)* vertexColor * intensity;
}
"""
vanish_shader = """
#version 460

layout (location = 0) out vec4 fragColor;

in float intensity;
in vec4 vertexColor;
in vec2 vertexTexcoords;
in vec3 lPosition;

uniform vec4 color;
uniform float time2;
uniform vec4 ambient;
uniform float rate;

uniform sampler2D tex;

void main()
{
    float amplitude = sin(time2)*sin(time2);
    float final_offset = (sin(lPosition.y * rate) * (amplitude));
    vec2 texCoord = vec2(vertexTexcoords.x + final_offset, vertexTexcoords.y);
    vec4 texColor = texture(tex, texCoord);
    texColor.a = clamp(texColor.a - (amplitude), 0.0, 1.0);
    fragColor =  texColor;
}
"""

shader1 = compileProgram(
    compileShader(vertex_shader, GL_VERTEX_SHADER),
    compileShader(fragment_shader, GL_FRAGMENT_SHADER),
)
shader2 = compileProgram(
    compileShader(vertex_shader, GL_VERTEX_SHADER),
    compileShader(rotation_shader, GL_FRAGMENT_SHADER),
)
shader3 =compileProgram(
    compileShader(vertex_shader, GL_VERTEX_SHADER),
    compileShader(position_shader, GL_FRAGMENT_SHADER),
)
shader4 = compileProgram(
    compileShader(vertex_shader, GL_VERTEX_SHADER),
    compileShader(vanish_shader, GL_FRAGMENT_SHADER),
)
shaders = [shader1, shader2, shader3, shader4]

scene = pyassimp.load('models/r2-d2.obj')

texture_surface = pygame.image.load('models/R2D2_Diffuse.jpg')
texture_data = pygame.image.tostring(texture_surface, 'RGB',1)
width = texture_surface.get_width()
height = texture_surface.get_height()

texture = glGenTextures(1)
glBindTexture(GL_TEXTURE_2D, texture)
glTexImage2D(
    GL_TEXTURE_2D,
    0,
    GL_RGB,
    width,
    height,
    0,
    GL_RGB,
    GL_UNSIGNED_BYTE,
    texture_data
)
glGenerateMipmap(GL_TEXTURE_2D)

def glize(node, counter1,counter2, shader):
    for mesh in node.meshes:
        vertex_data = numpy.hstack([
            numpy.array(mesh.vertices, dtype=numpy.float32),
            numpy.array(mesh.normals, dtype=numpy.float32),
            numpy.array(mesh.texturecoords[0], dtype=numpy.float32),
        ])

        index_data = numpy.hstack(
            numpy.array(mesh.faces, dtype = numpy.int32),
        )

        vertex_buffer_object = glGenVertexArrays(1)
        glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_object)
        glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9*4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9*4, ctypes.c_void_p(3*4))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 9*4, ctypes.c_void_p(6*4))
        glEnableVertexAttribArray(2)

        element_buffer_object = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, element_buffer_object)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, index_data.nbytes, index_data, GL_STATIC_DRAW)

        glUniform3f(glGetUniformLocation(shaders[shader % 4], "light"), 0, 0, 500)
        glUniform4f(glGetUniformLocation(shaders[shader % 4], "color"),0.75, 0.75, 0.75, 1)
        glUniform4f(glGetUniformLocation(shaders[shader % 4], "ambient"),0, 0, 0, 0)
        glUniform1f(glGetUniformLocation(shaders[shader % 4], "time1"), counter1)
        glUniform1f(glGetUniformLocation(shaders[shader % 4], "time2"), counter2)
        glUniform1f(glGetUniformLocation(shaders[shader % 4], "rate"), 5)


        glDrawElements(GL_TRIANGLES, len(index_data), GL_UNSIGNED_INT, None)

        

    for child in node.children:
        glize(child,counter1,counter2, shader)

i = glm.mat4()

def createTheMatrix(rotation, Cameraposition):

    translate = glm.translate(i, glm.vec3(0, -10, 0))
    rotate = glm.rotate(i,glm.radians(rotation), glm.vec3(0, 1, 0))
    scale = glm.scale(i, glm.vec3(0.5, 0.5, 0.5))

    model = translate * rotate * scale
    view = glm.lookAt(glm.vec3(0, 0, Cameraposition), glm.vec3(0, 25, 0), glm.vec3(0, 1, 0))
    projection = glm.perspective(glm.radians(45), 800/600, 0.1, 1000)

    return projection * view * model

glViewport(0,0,800,600)
glEnable(GL_DEPTH_TEST)

running = True
rotation = 0
time1 = 0
time2 = 0
counter = 0
position = 200
while running:
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glClearColor(0.5, 1.0, 0.5, 1.0)
    glUseProgram(shaders[counter % 4])

    theMatrix = createTheMatrix(rotation, position)

    theMatrixLocation = glGetUniformLocation(shaders[counter % 4], 'theMatrix')

    glUniformMatrix4fv(
        theMatrixLocation, #location
        1, #count
        GL_FALSE,
        glm.value_ptr(theMatrix) #pointer
    )
    
    glize(scene.rootnode, time1,time2,counter)
    pygame.display.flip()


    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                rotation -= 10
            if event.key == pygame.K_RIGHT:
                rotation += 10
            if event.key == pygame.K_UP and position > 30:
                position -= 10
            if event.key == pygame.K_DOWN:
                position += 10
            if event.key == pygame.K_SPACE:
                counter +=1
            if event.key == pygame.K_w:
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            if event.key == pygame.K_f:
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
    if counter % 4 ==3:
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    else:
        glDisable(GL_BLEND)
        
    time1 += 0.05
    time2 +=0.025
    clock.tick(60)