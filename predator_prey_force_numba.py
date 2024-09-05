import numba as nb
import numpy as np
from numba import njit, float64, int32

@njit
def calculate_forces_numba(positions, species, predator_species, prey_species, escape_distance, chase_distance, escape_weight, chase_weight):
    
    num_agents = len(positions)
    forces = np.zeros_like(positions)

    for current_species in range(1, 9):
        current_species_indices = np.where(species == current_species)[0]
        
        if len(current_species_indices) == 0:
            continue

        current_predator = predator_species[current_species - 1]
        predator_indices = np.where(species == current_predator)[0]

        if len(predator_indices) > 0:
            for agent_index in current_species_indices:
                agent_pos = positions[agent_index]
                predator_distances = np.sqrt(np.sum((positions[predator_indices] - agent_pos)**2, axis=1))
                
                close_predators = predator_indices[predator_distances < escape_distance]
                if len(close_predators) > 0:
                    nearest_predator = close_predators[np.argmin(predator_distances[predator_distances < escape_distance])]
                    escape_direction = agent_pos - positions[nearest_predator]
                    escape_direction = escape_direction / np.sqrt(np.sum(escape_direction**2))
                    forces[agent_index] += escape_direction * escape_weight

        current_prey = prey_species[current_species - 1]
        prey_indices = np.where(species == current_prey)[0]

        if len(prey_indices) > 0:
            for agent_index in current_species_indices:
                agent_pos = positions[agent_index]
                prey_distances = np.sqrt(np.sum((positions[prey_indices] - agent_pos)**2, axis=1))
                
                close_preys = prey_indices[prey_distances < chase_distance]
                if len(close_preys) > 0:
                    nearest_prey = close_preys[np.argmin(prey_distances[prey_distances < chase_distance])]
                    chase_direction = positions[nearest_prey] - agent_pos
                    chase_direction = chase_direction / np.sqrt(np.sum(chase_direction**2))
                    forces[agent_index] += chase_direction * chase_weight

    return forces

