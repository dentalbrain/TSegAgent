#version 450
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec4 color;
layout (location = 3) in int triangleId;

out vec4 VertexColor;
out vec3 FragPos;
out vec3 Normal;
out flat int TriangleId;

uniform bool vertexColors = false;
uniform vec4 vertexColor;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * model * vec4(position, 1.0);
    if (vertexColors) VertexColor = color; else VertexColor = vertexColor;
    FragPos = vec3(model * vec4(position, 1.0f));
    Normal = vec3(model * vec4(normal, 1.0f));
    TriangleId = triangleId;
}