#version 460
in vec4 VertexColor; // 从顶点着色器传来的输入变量（名称相同、类型相同）
in vec3 FragPos;
in vec3 Normal;

uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 lightColor;

out vec4 color;

void main()
{
    vec3 objectColor = vec3(VertexColor);
    vec3 norm = normalize(Normal);
    vec3 normNeg = -1 * norm;

    // Ambient
    float ambientStrength = 0.1f;
    vec3 ambient = ambientStrength * lightColor;

    // Diffuse
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor + max(dot(normNeg, lightDir), 0.0) * lightColor;

    // Specular
    float specularStrength = 0.5f;
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 reflectNegDir = reflect(-lightDir, normNeg);
    float specNeg = pow(max(dot(viewDir, reflectNegDir), 0.0), 32);
    vec3 specular = specularStrength * spec * lightColor + specularStrength * specNeg * lightColor;

    vec3 result = (ambient + diffuse + specular) * objectColor;
    color = vec4(result, 1.0f);
}