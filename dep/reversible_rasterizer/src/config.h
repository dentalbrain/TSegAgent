#ifndef REVERSIBLE_RASTERIZER_CONFIG_H
#define REVERSIBLE_RASTERIZER_CONFIG_H

#include <vector>

struct CameraConfig {
    int width;
    int height;
    std::vector<float> R;
    std::vector<float> T;
    double zoom;
    double near;
    double far;

    CameraConfig() {
        this->width = 1280;
        this->height = 800;
        this->R = {0.f, 0.f, 0.f};
        this->T = {0.f, 0.f, 0.f};
        this->zoom = 1;
        this->near = 0.0001;
        this->far = 10000;
    }
};

struct RenderConfig {
    bool vertexColors = false;
    std::vector<float> backgroundColor = {1.f, 1.f, 1.f};
    std::vector<float> vertexColor = {0.8f, 0.8f, 0.8f};
    float ambientStrength  = 0.3f;
    float diffuseStrength  = 0.5f;
    float specularStrength = 0.2f;
    std::vector<float> R = {0.f, 0.f, 0.f};
    std::vector<float> T = {0.f, 0.f, 0.f};
};

#endif //REVERSIBLE_RASTERIZER_CONFIG_H
