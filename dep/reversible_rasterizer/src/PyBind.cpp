#ifdef FOUND_PYBIND11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "api.h"
#include "mesh.hpp"

namespace py = pybind11;

PYBIND11_MODULE(librevras, m)
{
      m.doc() = "Reverse the rasterizer process, find the triangle ID through pixel.";

      py::class_<CameraConfig>(m, "CameraConfig")
          .def(py::init())
          .def_readwrite("width", &CameraConfig::width)
          .def_readwrite("height", &CameraConfig::height)
          .def_readwrite("R", &CameraConfig::R)
          .def_readwrite("T", &CameraConfig::T)
          .def_readwrite("zoom", &CameraConfig::zoom)
          .def_readwrite("near", &CameraConfig::near)
          .def_readwrite("far", &CameraConfig::far);

      py::class_<RenderConfig>(m, "RenderConfig")
          .def(py::init())
          .def_readwrite("vertexColors", &RenderConfig::vertexColors)
          .def_readwrite("vertexColor", &RenderConfig::vertexColor)
          .def_readwrite("backgroundColor", &RenderConfig::backgroundColor)
          .def_readwrite("ambientStrength", &RenderConfig::ambientStrength)
          .def_readwrite("diffuseStrength", &RenderConfig::diffuseStrength)
          .def_readwrite("specularStrength", &RenderConfig::specularStrength)
          .def_readwrite("R", &RenderConfig::R)
          .def_readwrite("T", &RenderConfig::T);

      py::class_<Mesh>(m, "Mesh")
          .def(py::init<std::vector<std::vector<double>>, std::vector<std::vector<double>>, std::vector<std::vector<uint32_t>>, std::vector<std::vector<float>>>())
          .def(py::init<std::vector<std::vector<double>>, std::vector<std::vector<double>>, std::vector<std::vector<uint32_t>>>());

      py::class_<Renderer>(m, "Renderer");

      m.def("init", &init, "Create renderer", py::return_value_policy::reference);

      m.def("setup_mesh", &setup_mesh, "Setup mesh before rendering",
            py::arg("mesh"),
            py::arg("config"));

      m.def("render_id_map", &render_id_map, "Render and get pixel-triangle map",
            py::arg("renderer"),
            py::arg("mesh"),
            py::arg("config"),
            py::arg("render_config"),
            py::return_value_policy::move);

      m.def("render", &render, "Render and get pixel-triangle map",
            py::arg("renderer"),
            py::arg("mesh"),
            py::arg("config"),
            py::arg("render_config"),
            py::return_value_policy::move);

      m.def("render_normal", &render_normal, "Render and get pixel-triangle map",
            py::arg("renderer"),
            py::arg("mesh"),
            py::arg("config"),
            py::arg("render_config"),
            py::return_value_policy::move);

      m.def("render_depth", &render_depth, "Render and get pixel-triangle map",
            py::arg("renderer"),
            py::arg("mesh"),
            py::arg("config"),
            py::arg("render_config"),
            py::return_value_policy::move);

      m.def("render_stacked", &render_stacked, "Render and get pixel-triangle map",
            py::arg("renderer"),
            py::arg("mesh"),
            py::arg("config"),
            py::arg("render_config"),
            py::arg("stack_mask"),
            py::arg("stack_alphas"),
            py::return_value_policy::move);

      m.def("end", &end, "Terminate the renderer", py::arg("renderer"));
      m.def("end_with_mesh", &end_with_mesh, "Terminate the renderer and mesh", py::arg("renderer"), py::arg("mesh"));
}
#endif
