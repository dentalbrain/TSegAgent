#version 450
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;

out vec3 FragPos;
out vec3 Normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * model * vec4(position, 1.0);
    FragPos = vec3(model * vec4(position, 1.0f));
    // 将法线转换到视图空间
    Normal = mat3(transpose(inverse(model))) * normal;
}
