#version 450
in vec3 FragPos;
in vec3 Normal;

out vec4 FragColor;

void main()
{
    // 将法线向量映射到 [0, 1] 范围用于可视化
    vec3 normalColor = normalize(Normal) * 0.5 + 0.5;
    FragColor = vec4(normalColor, 1.0);
}
