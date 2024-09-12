// predator_prey.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>  // Add this line for automatic STL conversions
#include <cmath>
#include <vector>

namespace py = pybind11;

std::vector<std::vector<float>> predator_prey_forces(
    py::array_t<float> positions,
    py::array_t<float> distances,
    py::array_t<int> species,
    py::array_t<int> predator_species,
    py::array_t<int> prey_species,
    float escape_distance,
    float chase_distance,
    float escape_weight,
    float chase_weight
) {
    auto positions_buf = positions.request();
    auto distances_buf = distances.request();
    auto species_buf = species.request();
    auto predator_species_buf = predator_species.request();
    auto prey_species_buf = prey_species.request();

    if (positions_buf.ndim != 2 || distances_buf.ndim != 2 || species_buf.ndim != 1 ||
        predator_species_buf.ndim != 1 || prey_species_buf.ndim != 1) {
        throw std::runtime_error("Number of dimensions must be 2 for positions and distances, 1 for others");
    }

    size_t num_agents = positions_buf.shape[0];
    float* pos_ptr = static_cast<float*>(positions_buf.ptr);
    float* dist_ptr = static_cast<float*>(distances_buf.ptr);
    int* species_ptr = static_cast<int*>(species_buf.ptr);
    int* predator_ptr = static_cast<int*>(predator_species_buf.ptr);
    int* prey_ptr = static_cast<int*>(prey_species_buf.ptr);

    std::vector<std::vector<float>> total_force(num_agents, std::vector<float>(2, 0.0f));

    for (size_t i = 0; i < num_agents; ++i) {
        float nearest_predator_dist = std::numeric_limits<float>::max();
        float nearest_prey_dist = std::numeric_limits<float>::max();
        size_t nearest_predator_idx = 0;
        size_t nearest_prey_idx = 0;

        for (size_t j = 0; j < num_agents; ++j) {
            if (i == j) continue;

            float dist = dist_ptr[i * num_agents + j];
            if (dist < escape_distance && species_ptr[j] == predator_ptr[species_ptr[i] - 1]) {
                if (dist < nearest_predator_dist) {
                    nearest_predator_dist = dist;
                    nearest_predator_idx = j;
                }
            }
            if (dist < chase_distance && species_ptr[j] == prey_ptr[species_ptr[i] - 1]) {
                if (dist < nearest_prey_dist) {
                    nearest_prey_dist = dist;
                    nearest_prey_idx = j;
                }
            }
        }

        if (nearest_predator_dist < escape_distance) {
            float dx = pos_ptr[i * 2] - pos_ptr[nearest_predator_idx * 2];
            float dy = pos_ptr[i * 2 + 1] - pos_ptr[nearest_predator_idx * 2 + 1];
            float magnitude = std::sqrt(dx * dx + dy * dy);
            total_force[i][0] += escape_weight * dx / magnitude;
            total_force[i][1] += escape_weight * dy / magnitude;
        }

        if (nearest_prey_dist < chase_distance) {
            float dx = pos_ptr[nearest_prey_idx * 2] - pos_ptr[i * 2];
            float dy = pos_ptr[nearest_prey_idx * 2 + 1] - pos_ptr[i * 2 + 1];
            float magnitude = std::sqrt(dx * dx + dy * dy);
            total_force[i][0] += chase_weight * dx / magnitude;
            total_force[i][1] += chase_weight * dy / magnitude;
        }
    }

    return total_force;
}

PYBIND11_MODULE(predator_prey_module, m) {
    m.def("predator_prey_forces", &predator_prey_forces, "Calculate predator-prey forces");
}