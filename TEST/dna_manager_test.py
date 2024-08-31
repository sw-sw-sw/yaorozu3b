from dna_manager import DNAManager

def test_dna_manager():
    dna_manager = DNAManager()
    traits = [
        "INITIAL_AGENT_NUM", "MAX_FORCE", "ESCAPE_DISTANCE", "ESCAPE_WEIGHT",
        "CHASE_DISTANCE", "CHASE_WEIGHT", "MASS", "FRICTION", "DENSITY",
        "RESTITUTION", "DAMPING","LIFE_ENERGY", "LIFE_ENERGY_LOSS_RATE",
        "ENERGY_GAIN_ON_CONTACT", "BIRTH_THRESHOLD", "REPRODUCTION_RATE",
        "SIZE", "SPEED", "HORN_NUM", "HORN_LENGTH", "HORN_WIDTH",
        "SHELL_SIZE", "SHELL_POINT_SIZE", "COLOR", "PREDATOR_SPECIES",
        "PREDATOR_RATE", "PREY_SPECIES"
    ]

    print("DNA Information for Species 1-8:")
    print("-" * 50)

    for species in range(1, 9):
        print(f"Species {species}:")

        for trait in traits:
            try:
                value = dna_manager[species].get_trait_value(trait)
                print(f"  {trait}: {value}")
            except KeyError:
                print(f"  {trait}: Not defined")

        print("-" * 50)

    print("\nTrait Ranges:")
    print("-" * 50)
    for trait in traits:
        try:
            min_val, max_val = dna_manager.get_trait_range(trait)
            print(f"{trait}: Min = {min_val}, Max = {max_val}")
        except KeyError:
            print(f"{trait}: Range not defined")

if __name__ == "__main__":
    test_dna_manager()