#version 450
in vec4 VertexColor; // 从顶点着色器传来的输入变量（名称相同、类型相同）
in vec3 FragPos;
in vec3 Normal;
in flat int TriangleId;

uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 lightColor;

uniform float uAmbientStrength  = 0.3f;
uniform float uDiffuseStrength  = 0.5f;
uniform float uSpecularStrength = 0.2f;

layout(location = 0) out vec4 color;
layout(location = 1) out int fragmentID; // 输出到颜色附件的 ID

void main()
{
    vec3 objectColor = vec3(VertexColor);
    vec3 norm = normalize(Normal);
    vec3 normNeg = -1 * norm;

    // Ambient
    vec3 ambient = uAmbientStrength * lightColor;

    // Diffuse
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = uDiffuseStrength * diff * lightColor
                 + uDiffuseStrength * max(dot(normNeg, lightDir), 0.0) * lightColor;

    // Specular
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 reflectNegDir = reflect(-lightDir, normNeg);
    float specNeg = pow(max(dot(viewDir, reflectNegDir), 0.0), 32);
    vec3 specular = uSpecularStrength * spec * lightColor
                  + uSpecularStrength * specNeg * lightColor;

    vec3 result = (ambient + diffuse + specular) * objectColor;
    color = vec4(result, 1.0f);

    fragmentID = TriangleId;
    if (!gl_FrontFacing) {
        fragmentID = -1;
        color = vec4(0.15, 0.15, 0.15, 1.0f);
    }
}
