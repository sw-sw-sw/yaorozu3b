import unittest
from config_manager import ConfigManager, DNASpecies

class TestConfigManager(unittest.TestCase):
    def setUp(self):
        self.config_manager = ConfigManager()

    def test_singleton(self):
        another_instance = ConfigManager()
        self.assertIs(self.config_manager, another_instance)

    def test_load_config(self):
        self.assertTrue(len(self.config_manager.config) > 0)

    def test_get_trait_value(self):
        world_width = self.config_manager.get_trait_value('WORLD_WIDTH')
        self.assertEqual(world_width, 2000)

    def test_get_trait_range(self):
        min_val, max_val = self.config_manager.get_trait_range('WORLD_WIDTH')
        self.assertEqual(min_val, 1000)
        self.assertEqual(max_val, 10000)

    def test_get_species_trait_value(self):
        mass_species_1 = self.config_manager.get_species_trait_value('MASS', 1)
        self.assertEqual(mass_species_1, 1.0)
        
        # GLOBALとSpecies固有の値の比較
        global_mass = self.config_manager.get_trait_value('MASS')
        self.assertNotEqual(mass_species_1, global_mass)

    def test_species_specific_override(self):
        global_size = self.config_manager.get_trait_value('SIZE')
        species_3_size = self.config_manager.get_species_trait_value('SIZE', 3)
        self.assertNotEqual(global_size, species_3_size)
        self.assertEqual(species_3_size, 25)  # config.csvの値に基づいて

    def test_global_fallback(self):
        # Species固有の値が設定されていない場合のテスト
        global_dt = self.config_manager.get_trait_value('DT')
        species_dt = self.config_manager.get_species_trait_value('DT', 1)
        self.assertEqual(global_dt, species_dt)

    def test_get_dna_for_species(self):
        dna_species_1 = self.config_manager.get_dna_for_species(1)
        self.assertIsInstance(dna_species_1, DNASpecies)
        self.assertEqual(dna_species_1.species_id, 1)
        self.assertEqual(dna_species_1.get_trait('MASS'), 1.0)

    def test_dna_species_consistency(self):
        dna_species_1 = self.config_manager.get_dna_for_species(1)
        mass_from_dna = dna_species_1.get_trait('MASS')
        mass_from_config = self.config_manager.get_species_trait_value('MASS', 1)
        self.assertEqual(mass_from_dna, mass_from_config)

    def test_invalid_species(self):
        with self.assertRaises(KeyError):
            self.config_manager.get_species_trait_value('MASS', 10)  # 存在しない種

    def test_invalid_trait(self):
        with self.assertRaises(KeyError):
            self.config_manager.get_trait_value('INVALID_TRAIT')

    def test_display_methods(self):
        # This test just ensures that the display methods run without errors
        try:
            self.config_manager.display_environment_variables()
            self.config_manager.display_species_dna()
        except Exception as e:
            self.fail(f"display methods raised {type(e).__name__} unexpectedly!")

if __name__ == '__main__':
    config_manager = ConfigManager()
    print("Displaying Environment Variables:")
    config_manager.display_environment_variables()
    print("\nDisplaying Species DNA:")
    config_manager.display_species_dna()
    
    unittest.main()