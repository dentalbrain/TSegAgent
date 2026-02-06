#version 450

// 顶点位置输入
layout (location = 0) in vec3 position;

// 统一变量，用于矩阵变换
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

// 输出视图空间下的顶点位置
out vec4 viewSpacePos;

void main()
{
    // 计算模型 - 视图矩阵
    vec4 modelPos = projection * view * model * vec4(position, 1.0);
    // 将顶点位置转换到视图空间
//    viewSpacePos = vec3(modelView * vec4(position, 1.0));
    viewSpacePos = modelPos;
    // 计算裁剪空间下的顶点位置
    gl_Position = modelPos;
}