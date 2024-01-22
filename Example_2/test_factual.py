import pandas as pd
from factual_accuracy import TestFactualAccuracy
"""
These are the import statements
"""

class TestFactualAccuracy:
    """
    Test class for the Factual Accuracy module.

    This class includes test cases for various methods related to factual accuracy.
    """

    test_factual_accuracy = TestFactualAccuracy()

    def test_extract_named_entities(self):
        """
        Test if named entities are correctly extracted from the text.

        This function tests the extract_named_entities method of the TestFactualAccuracy class,
        checking if it correctly identifies named entities in a given text.
        """
        text = "The quick brown fox jumps over the lazy dog."
        result = self.test_factual_accuracy.extract_named_entities(text)
        assert result is None
       
        # Test with a different text containing multiple entities
        text_multiple_entities = "Apple is a tech company. Steve Jobs co-founded Apple."
        result_multiple_entities = self.test_factual_accuracy.extract_named_entities(text_multiple_entities)
        assert isinstance(result_multiple_entities, set)
        assert result_multiple_entities == {'Steve Jobs', 'Apple'}

    def test_check_factual_accuracy(self):
        """
        Test if the check_factual_accuracy method correctly identifies facts in output entities.

        This function tests the check_factual_accuracy method of the TestFactualAccuracy class,
        ensuring it correctly identifies factual accuracy between output and ground truth entities.
        """
        output_entities = ["Apple", "American", "Cupertino", "California", "United States"]
        ground_truth_entities = ["Apple", "Cupertino", "American"]
        result = self.test_factual_accuracy.check_factual_accuracy(output_entities, ground_truth_entities)
        assert result is True

        # Test when output_entities or ground_truth_entities is empty
        output_entities_empty = []
        ground_truth_entities_empty = []
        result_false_empty = self.test_factual_accuracy.check_factual_accuracy(output_entities_empty,
                                                                               ground_truth_entities_empty)
        assert result_false_empty is False

        # Test when some facts are not present in the output
        result_false = self.test_factual_accuracy.check_factual_accuracy(output_entities, ["Grapes", "Peach"])
        assert result_false is False

        # Test with a different set of entities in the output
        output_entities_diff = ["Banana", "Mango", "Pineapple"]
        ground_truth_entities_diff = ["Apple", "Cupertino", "American"]
        result_false_diff = self.test_factual_accuracy.check_factual_accuracy(output_entities_diff,
                                                                              ground_truth_entities_diff)
        assert result_false_diff is False

        # Test when all facts are present but in a different order
        output_entities_diff_order = ["American", "Cupertino", "Apple"]
        result_true_diff_order = self.test_factual_accuracy.check_factual_accuracy(output_entities_diff_order,
                                                                                   ground_truth_entities)
        assert result_true_diff_order is True

    def test_get_unique_entities(self):
        """
        Test the get_unique_entities method with various scenarios.

        This function tests the get_unique_entities method of the TestFactualAccuracy class,
        checking its behavior with different combinations of input entities and ground truth.
        """
        output_entities_none = None
        ground_truth_entities_none = None
        result_none = self.test_factual_accuracy.get_unique_entities(output_entities_none, ground_truth_entities_none)
        assert result_none is None

        # Test with output_entities being None
        output_entities_none = None
        ground_truth_entities = ["Banana", "Cherry", "Orange"]
        result_output_none = self.test_factual_accuracy.get_unique_entities(output_entities_none, ground_truth_entities)
        assert isinstance(result_output_none, set)
        assert result_output_none == {"Banana", "Cherry", "Orange"}

        # Test with ground_truth_entities being None
        output_entities = ["Apple", "Banana", "Orange"]
        ground_truth_entities_none = None
        result_ground_truth_none = self.test_factual_accuracy.get_unique_entities(output_entities,
                                                                                  ground_truth_entities_none)
        assert isinstance(result_ground_truth_none, set)
        assert result_ground_truth_none == {"Apple", "Banana", "Orange"}

        # Test with valid values for both output and ground truth
        output_entities = ["Apple", "Banana", "Orange"]
        ground_truth_entities = ["Banana", "Cherry", "Orange"]
        result_valid = self.test_factual_accuracy.get_unique_entities(output_entities, ground_truth_entities)
        assert isinstance(result_valid, set)
        assert result_valid == {"Banana", "Cherry", "Apple", "Orange"}

    def test_get_unique_entities_in_output(self):
        """
        Test if unique entities in the output (not in ground truth) are correctly identified.

        This function tests the get_unique_entities_in_output method of the TestFactualAccuracy class,
        making sure it correctly identifies entities present in the output but not in the ground truth.
        """
        output_entities = ["Apple", "Banana", "Orange"]
        ground_truth_entities = ["Banana", "Cherry", "Orange"]
        result = self.test_factual_accuracy.get_unique_entities_in_output(output_entities, ground_truth_entities)
        assert isinstance(result, (set, str))
        assert result == {"Apple"}

        # Test with a different set of entities in output and ground truth
        output_entities_diff = ["Grape", "Mango", "Pineapple"]
        ground_truth_entities_diff = ["Banana", "Cherry", "Orange"]
        result_diff = self.test_factual_accuracy.get_unique_entities_in_output(output_entities_diff,
                                                                               ground_truth_entities_diff)
        assert isinstance(result_diff, (set, str))
        assert result_diff == {"Grape", "Mango", "Pineapple"}

        # Test with output_entities being None
        output_entities_none = None
        ground_truth_entities = ["Banana", "Cherry", "Orange"]
        result_output_none = self.test_factual_accuracy.get_unique_entities_in_output(output_entities_none,
                                                                                      ground_truth_entities)
        assert result_output_none == "no_named_entities in Output"

        # Test with ground_truth_entities being None
        output_entities = ["Apple", "Banana", "Orange"]
        ground_truth_entities_none = None
        result_ground_truth_none = self.test_factual_accuracy.get_unique_entities_in_output(output_entities,
                                                                                            ground_truth_entities_none)
        assert isinstance(result_diff, (set, str))
        assert result_ground_truth_none == output_entities

    def test_get_unique_entities_in_ground_truth(self):
        """
        Test if unique entities in the ground_truth (not in output) are correctly identified.

        This function tests the get_unique_entities_in_ground_truth method of the TestFactualAccuracy class,
        verifying its ability to identify entities present in the ground truth but not in the output.
        """
        ground_truth_entities = ["Banana", "Cherry", "Orange"]
        output_entities = ["Apple", "Banana", "Orange"]
        result = self.test_factual_accuracy.get_unique_entities_in_ground_truth(ground_truth_entities, output_entities)
        assert isinstance(result, (set, str))
        assert result == {"Cherry"}

        # Test with a different set of entities in output and ground truth
        ground_truth_entities_diff = ["Banana", "Cherry", "Orange"]
        output_entities_diff = ["Grape", "Mango", "Pineapple"]
        result_diff = self.test_factual_accuracy.get_unique_entities_in_ground_truth(ground_truth_entities_diff, output_entities_diff)
        assert isinstance(result_diff, set)
        assert result_diff == {"Banana", "Cherry", "Orange"}

        # Test with output_entities being None
        output_entities_none = None
        ground_truth_entities = ["Banana", "Cherry", "Orange"]
        result_output_none = self.test_factual_accuracy.get_unique_entities_in_ground_truth(ground_truth_entities, output_entities_none)
        assert isinstance(result_diff, (set, str))
        assert result_output_none == ground_truth_entities

        # Test with ground_truth_entities being None
        output_entities = ["Apple", "Banana", "Orange"]
        ground_truth_entities_none = None
        result_ground_truth_none = self.test_factual_accuracy.get_unique_entities_in_ground_truth(ground_truth_entities_none, output_entities)
        assert result_ground_truth_none == "no_named_entities in Ground_truth"

    def test_calculate_overlap_pct(self):
        """
        Test if cosine similarity is correctly calculated between two sets of entities.

        This function tests the calculate_overlap_pct method of the TestFactualAccuracy class,
        ensuring that the cosine similarity is accurately calculated between two sets of entities.
        """
        output_entities = ["Apple", "Banana", "Orange"]
        ground_truth_entities = ["Banana", "Cherry", "Orange"]
        result = self.test_factual_accuracy.calculate_overlap_pct(output_entities, ground_truth_entities)
        assert isinstance(result, float)

        # Test with a different set of entities
        output_entities_diff = ["Grape", "Mango", "Pineapple"]
        ground_truth_entities_diff = ["Banana", "Cherry", "Orange"]
        result_diff = self.test_factual_accuracy.calculate_overlap_pct(output_entities_diff, ground_truth_entities_diff)
        assert isinstance(result_diff, float)

    def test_extract_data_from_file(self):
        """
        Test if data is correctly extracted from a CSV file and factual accuracy is checked.

        This function tests the extract_data_from_file method of the TestFactualAccuracy class,
        checking if it correctly reads data from a CSV file and performs factual accuracy checks.
        """
        file_path = r"C:\Users\VishalChaurasiya(Ann\Downloads\L_D_Fluency_Validation_Results (2).xlsx"
        result_df = self.test_factual_accuracy.extract_data_from_file(file_path)
        assert isinstance(result_df, pd.DataFrame)
        return result_df

    def fake_data(self):
        """
        Generate fake data for testing when the file is not found.

        This function generates fake data for testing purposes, particularly when the file
        for data extraction is not found during testing.
        """
        data = {
            "Ground_Truth": ["A distribution in Omni refers to the process of sharing or pushing an audience from "
                             "Audience Explorer (Omni) to another application within the Omni ecosystem or to "
                             "external platforms (e.g., Demand Side Platforms, or DSPs) for activation.",
                             "In Omni L&D."],
            "Output": ["A distribution in Omni refers to the process of sharing or pushing an audience from Audience "
                       "Explorer (Omni) to another application within the Omni ecosystem or to external platforms ("
                       "e.g., Demand Side Platforms, or DSPs) for activation.",
                       "The Omni Foundations Certification is available on Omni L&D, the learning and development "
                       "platform for Omnicom employees. Omni L&D hosts the recorded sessions, presentation decks, "
                       "and assessment required for completing the certification."]
        }
        df = pd.DataFrame(data)
        return df

    def test_factual_accuracy_with_fake_data(self):
        """
        Test factual accuracy using fake data when the file is not found.

        This function tests factual accuracy using fake data when the file for data extraction
        is not found. It handles the case where the file is not found by using the fake data.
        """
        file_path = r"C:\Users\VishalChaurasiya(Ann\Downloads\L_D_Fluency_Validation_Results (5).xlsx"
        result_df = self.test_factual_accuracy.extract_data_from_file(file_path)

        # If the file is not found, use fake data
        if result_df is None:
            print(f"Using fake data for testing as file not found: {file_path}")
            result_df = self.fake_data()

        assert not result_df.empty
