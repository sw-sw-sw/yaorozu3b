import csv
from typing import Dict, Any, Tuple

class DNASpecies:
    def __init__(self, species_id: int, traits: Dict[str, Any]):
        self.species_id = species_id
        self.traits = traits

    def get_trait(self, trait_name: str) -> Any:
        return self.traits.get(trait_name)
    
class ConfigManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, file_path: str = 'config.csv'):
        if self._initialized:
            return
        self._initialized = True
        self.file_path = file_path
        self.config: Dict[str, Any] = {}
        self.species_dna: Dict[int, DNASpecies] = {}
        self.load_config()

    def load_config(self):
        try:
            with open(self.file_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    trait = row['TRAIT_NAME']
                    if trait:  # Skip empty rows
                        self.config[trait] = {key: value for key, value in row.items() if value}
                        
                        # Initialize species DNA
                        for species_id in range(1, 9):
                            if str(species_id) not in self.config[trait]:
                                self.config[trait][str(species_id)] = self.config[trait].get('GLOBAL')

        except FileNotFoundError:
            raise FileNotFoundError(f"設定ファイルが見つかりません: {self.file_path}")
        except csv.Error as e:
            raise ValueError(f"CSVファイルの読み込み中にエラーが発生しました: {e}")

        # Create DNASpecies instances
        for species_id in range(1, 9):
            species_traits = {trait: self._parse_value(values[str(species_id)]) 
                              for trait, values in self.config.items() 
                              if str(species_id) in values}
            self.species_dna[species_id] = DNASpecies(species_id, species_traits)

    def _parse_value(self, value: str) -> Any:
        if value is None or value == '':
            return None
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value

    def get_trait_value(self, trait: str) -> Any:
        if trait not in self.config:
            raise KeyError(f"指定されたトレイト {trait} が見つかりません。")
        return self._parse_value(self.config[trait].get('GLOBAL'))

    def get_species_trait_value(self, trait: str, species: int) -> Any:
        if species not in self.species_dna:
            raise KeyError(f"指定された種 {species} が見つかりません。")
        return self.species_dna[species].get_trait(trait)

    def get_dna_for_species(self, species: int) -> DNASpecies:
        if species not in self.species_dna:
            raise KeyError(f"指定された種 {species} が見つかりません。")
        return self.species_dna[species]

    def get_trait_range(self, trait: str) -> Tuple[float, float]:
        if trait not in self.config:
            raise KeyError(f"指定されたトレイト {trait} が見つかりません。")
        trait_config = self.config[trait]
        min_value = self._parse_value(trait_config.get('Min', float('-inf')))
        max_value = self._parse_value(trait_config.get('Max', float('inf')))
        return min_value, max_value

    def display_environment_variables(self):
        print("Environment Variables:")
        for trait, values in self.config.items():
            if 'GLOBAL' in values:
                print(f"{trait}: {values['GLOBAL']} (Min: {values.get('Min', 'N/A')}, Max: {values.get('Max', 'N/A')})")

    def display_species_dna(self):
        print("\nSpecies DNA Values:")
        for species, dna in self.species_dna.items():
            print(f"\nSpecies {species}:")
            for trait, value in dna.traits.items():
                print(f"  {trait}: {value}")