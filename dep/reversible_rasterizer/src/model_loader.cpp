#include "model_loader.h"
#include <fstream>

ModelLoader::ModelLoader(const std::string &model_path) : model_path(model_path) {
    g_scene = importer.ReadFile(model_path, aiProcess_CalcTangentSpace | aiProcess_GenSmoothNormals | aiProcess_Triangulate | aiProcess_JoinIdenticalVertices);
}

ModelLoader::~ModelLoader() {
    // delete g_scene;
}

std::vector<std::vector<uint32_t>> ModelLoader::get_faces() {
    std::vector<std::vector<uint32_t>> faces;
    unsigned int add = 0;
    for (int i = 0; i < g_scene->mNumMeshes; i++) {
        aiMesh* mesh = g_scene->mMeshes[i];
        for (int fi = 0; fi < mesh->mNumFaces; fi++) {
            unsigned int* indices = mesh->mFaces[fi].mIndices;

            faces.push_back({
                indices[0] + add,
                indices[1] + add,
                indices[2] + add
            });
        }
        add += mesh->mNumVertices;
    }
    return faces;
}

std::vector<std::vector<double>> ModelLoader::get_vertices() {
    std::vector<std::vector<double>> vertices;
    for (int i = 0; i < g_scene->mNumMeshes; i++) {
        aiMesh* mesh = g_scene->mMeshes[i];
        for (int vi = 0; vi < mesh->mNumVertices; vi++) {
            aiVector3d vert = mesh->mVertices[vi];
            vertices.push_back({vert.x, vert.y, vert.z});
        }
    }
    return vertices;
}

std::vector<std::vector<double>> ModelLoader::get_normals() {
    std::vector<std::vector<double>> normals;
    for (int i = 0; i < g_scene->mNumMeshes; i++) {
        aiMesh* mesh = g_scene->mMeshes[i];
        for (int vi = 0; vi < mesh->mNumVertices; vi++) {
            aiVector3d vert = mesh->mNormals[vi];
            normals.push_back({vert.x, vert.y, vert.z});
        }
    }
    return normals;
}

std::vector<std::vector<float>> ModelLoader::get_vertex_colors() {
    std::vector<std::vector<float>> vc;
    for (int i = 0; i < g_scene->mNumMeshes; i++) {
        aiMesh* mesh = g_scene->mMeshes[i];
        aiColor4D * colors = mesh->mColors[0];
        for (int vi = 0; vi < mesh->mNumVertices; vi++) {
            if (colors != nullptr) {
                aiColor4D color = mesh->mColors[0][vi];
                vc.push_back({color.r, color.g, color.b, color.a});
            } else {
                vc.push_back({0.65, 0.65, 0.65, 1.0});
            }
        }
    }
    return vc;
}