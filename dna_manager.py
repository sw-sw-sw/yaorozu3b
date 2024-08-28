import csv
from typing import Dict, Any, Optional

class DNAManager:
    def __init__(self, file_path: str = 'dna_config.csv'):
        self.file_path = file_path
        self.config: Dict[str, Any] = {}
        self.load_config()

    def load_config(self):
        try:
            with open(self.file_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    trait = row['SPECIES_TYPE']
                    self._process_row(row, trait)
        except FileNotFoundError:
            raise FileNotFoundError(f"DNA設定ファイルが見つかりません: {self.file_path}")
        except csv.Error as e:
            raise ValueError(f"CSVファイルの読み込み中にエラーが発生しました: {e}")

    def _process_row(self, row: Dict[str, str], trait: str):
        self.config[trait] = {}
        if row['GLOBAL']:
            self.config[trait]['GLOBAL'] = self._parse_value(row['GLOBAL'])
        for species in range(1, 9):
            if row[str(species)]:
                self.config[trait][species] = self._parse_value(row[str(species)])
        for limit in ['Min', 'Max']:
            if row[limit]:
                self.config[trait][limit] = self._parse_value(row[limit])

    @staticmethod
    def _parse_value(value: str) -> Any:
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value

    def get_trait_value(self, trait: str, species: Optional[int] = None) -> Any:
        if trait not in self.config:
            raise KeyError(f"指定されたトレイト {trait} が見つかりません。")
        
        trait_config = self.config[trait]
        
        if species is not None:
            if species not in trait_config:
                return trait_config.get('GLOBAL')
            return trait_config[species]
        
        return trait_config.get('GLOBAL', next(iter(trait_config.values())))

    def get_trait_range(self, trait: str) -> tuple:
        if trait not in self.config:
            raise KeyError(f"指定されたトレイト {trait} が見つかりません。")
        
        trait_config = self.config[trait]
        min_value = trait_config.get('Min', float('-inf'))
        max_value = trait_config.get('Max', float('inf'))
        
        return (min_value, max_value)

    def __getitem__(self, species: int):
        return DNASpecies(self, species)

class DNASpecies:
    def __init__(self, dna_manager: DNAManager, species: int):
        self.dna_manager = dna_manager
        self.species = species

    def get_trait_value(self, trait: str) -> Any:
        return self.dna_manager.get_trait_value(trait, self.species)