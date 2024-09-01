import csv
from typing import Dict, Any, Optional

class DNAManager:
    def __init__(self, file_path: str = 'dna_config.csv'):
        self.file_path = file_path
        self.config: Dict[str, Any] = {}
        self.cache: Dict[str, Any] = {}  # キャッシュ用の辞書を追加
        self.load_config()

    def load_config(self):
        try:
            with open(self.file_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    trait = row['SPECIES_TYPE']
                    if trait:  # Skip empty rows
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
        # キャッシュキーを作成
        cache_key = f"{trait}_{species}" if species is not None else trait

        # キャッシュにある場合はキャッシュから返す
        if cache_key in self.cache:
            return self.cache[cache_key]

        if trait not in self.config:
            raise KeyError(f"指定されたトレイト {trait} が見つかりません。")

        trait_config = self.config[trait]

        if species is not None:
            value = trait_config.get(species, trait_config.get('GLOBAL'))
        else:
            value = trait_config.get('GLOBAL', next(iter(trait_config.values())))
        # 結果をキャッシュに保存
        self.cache[cache_key] = value

        return value

    def clear_cache(self):
        """キャッシュをクリアするメソッド"""
        self.cache.clear()

    def update_config(self, trait: str, species: Optional[int], value: Any):
        """設定を更新し、関連するキャッシュをクリアするメソッド"""
        if trait not in self.config:
            raise KeyError(f"指定されたトレイト {trait} が見つかりません。")

        if species is not None:
            self.config[trait][species] = value
        else:
            self.config[trait]['GLOBAL'] = value

        # 関連するキャッシュをクリア
        cache_key = f"{trait}_{species}" if species is not None else trait
        self.cache.pop(cache_key, None)
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