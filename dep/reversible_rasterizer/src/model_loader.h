#ifndef REVERSIBLE_RASTERIZER_MODEL_LOADER_H
#define REVERSIBLE_RASTERIZER_MODEL_LOADER_H

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <assimp/DefaultLogger.hpp>
#include <assimp/LogStream.hpp>
#include <stdfloat>

class ModelLoader {
public:
    const std::string model_path;
    const aiScene* g_scene = nullptr;

    explicit ModelLoader(const std::string &model_path);

    virtual ~ModelLoader();

    std::vector<std::vector<double>> get_vertices();
    std::vector<std::vector<double>> get_normals();
    std::vector<std::vector<uint32_t>> get_faces();
    std::vector<std::vector<float>> get_vertex_colors();

private:
    Assimp::Importer importer;
};


#endif //REVERSIBLE_RASTERIZER_MODEL_LOADER_H
