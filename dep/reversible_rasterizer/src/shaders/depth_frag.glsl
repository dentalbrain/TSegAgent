#version 450

in vec4 viewSpacePos;
out vec4 FragColor;

uniform float minDepth;
uniform float maxDepth;

void main()
{
    float viewDepth = viewSpacePos.z / viewSpacePos.w;
    // 将深度值归一化到 [0, 1] 范围
    float normalizedDepth = (viewDepth - minDepth) / (maxDepth - minDepth);
    // 限制深度值在 [0, 1] 范围内
    normalizedDepth = clamp(normalizedDepth, 0.0, 1.0);
    normalizedDepth = normalizedDepth * 0.9 + 0.05;
    // 将归一化后的深度值映射为灰度颜色
    FragColor = vec4(vec3(normalizedDepth), 1.0);
}