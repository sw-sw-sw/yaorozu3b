from dna_manager import DNAManager

def test_dna_manager():
    dna_manager = DNAManager()
    traits = [
        "SIZE", "SPEED", "HORN_NUM", "HORN_LENGTH", "HORN_WIDTH",
        "SHELL_SIZE", "SHELL_POINT_SIZE", "COLOR", "REPRODUCTION_THRESHOLD",
        "LIFESPAN", "TROPHIC_LEVEL", "PREDATION_EFFICIENCY"
    ]

    print("DNA Information for Species 1-8:")
    print("-" * 50)

    for species in range(1, 9):
        species_str = str(species)
        print(f"Species {species_str}:")
        
        for trait in traits:
            try:
                value = dna_manager[species_str].get_trait_value(trait)
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