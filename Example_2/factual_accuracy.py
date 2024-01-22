import spacy
import pandas as pd
from typing import List, Any
from sentence_transformers import SentenceTransformer, util
"""
These are the import statements

"""

class TestFactualAccuracy:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def extract_named_entities(self, text: str):
        """
        Fetch named entities from the provided text.
        Parameters:
        - text (str): The input text.
        Returns:
        - list: List of named entities.
        """
        doc = self.nlp(text)
        named_entities = set(ent.text for ent in doc.ents)

        if not named_entities:
            return None

        return named_entities

    def check_factual_accuracy(self, output_entities: list, ground_truth_entities: list) -> bool:
        """
        Check if the given ground_truth_entities are present in the output_entities.
        Parameters:
        - output_entities (list): List of named entities from the output.
        - ground_truth_entities (list): List of facts to check for in the output named entities.
        Returns:
        - bool: True if all facts are present, otherwise False.
        """
        if not ground_truth_entities or output_entities is None:
            return False

        for fact in ground_truth_entities:
            if not any(fact in entity for entity in output_entities):
                return False

        return True

    def get_unique_entities(self, output_entities: list, ground_truth_entities: list) -> None | set[Any]:
        """
        Return a unique list of entities found in both output_entities and ground_truth_entities.
        Parameters:
        - output_entities (list): List of named entities from the output.
        - ground_truth_entities (list): List of ground truth named entities.
        Returns:
        - list: Unique list of found entities.
        """
        if output_entities is None and ground_truth_entities is None:
            return None

        if output_entities is None:
            return set(ground_truth_entities)

        if ground_truth_entities is None:
            return set(output_entities)

        unique_entities = set(output_entities) | set(ground_truth_entities)
        return set(unique_entities)

    def get_unique_entities_in_output(self, output_entities: List[str], ground_truth_entities: List[str]) -> str | list[
        str] | set[str]:
        """
        Return the unique set of entities found in output_entities but not in ground_truth_entities.
        Parameters:
        - output_entities (list): List of named entities from the output.
        - ground_truth_entities (list): List of ground truth named entities.
        Returns:
        - set: Unique set of entities found in output_entities but not in ground_truth_entities.
        """
        if output_entities is None:
            return "no_named_entities in Output"

        if ground_truth_entities is None:
            return output_entities

        if output_entities == ground_truth_entities:
            return "all_attested"

        unique_set_in_output = set(output_entities) - set(ground_truth_entities)
        if not unique_set_in_output:
            return "No_Unique_Entities in Output"
        return unique_set_in_output

    def get_unique_entities_in_ground_truth(self, ground_truth_entities: List[str], output_entities: List[str]) -> list[str] | str | set[str]:
        """
        Return the unique set of entities found in ground_truth_entities but not in output_entities.
        Parameters:
        - ground_truth_entities (list): List of ground truth named entities.
        - output_entities (list): List of named entities from the output.
        Returns:
        - set: Unique set of entities found in ground_truth_entities but not in output_entities.
        """
        if output_entities is None:
            return ground_truth_entities

        if ground_truth_entities is None:
            return "no_named_entities in Ground_truth"

        if output_entities == ground_truth_entities:
            return "all_attested"

        unique_set_in_ground_truth = set(ground_truth_entities) - set(output_entities)
        if not unique_set_in_ground_truth:
            return "No_Unique_entities in Ground_truth"
        return unique_set_in_ground_truth

    def calculate_overlap_pct(self, output_entities: list, ground_truth_entities: list) -> float:
        """
        Calculate the overlap percentage between output_entities and ground_truth_entities.
        Parameters:
        - output_entities (list): List of named entities from the output.
        - ground_truth_entities (list): List of ground truth named entities.
        Returns:
        - float: Overlap percentage between the two sets of entities.
        """
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # You can choose a different model

        # Encode the entities into embeddings
        output_embeddings = model.encode(output_entities, convert_to_tensor=True)
        ground_truth_embeddings = model.encode(ground_truth_entities, convert_to_tensor=True)

        # Ensure both output and ground truth have the same number of vectors
        assert len(output_embeddings) == len(ground_truth_embeddings), "Mismatch in the number of vectors."

        # Calculate cosine similarity for each pair of vectors
        similarity_scores = util.pytorch_cos_sim(output_embeddings, ground_truth_embeddings)

        # Take the average similarity score across all pairs
        average_similarity = similarity_scores.mean().item()

        # Convert similarity score to percentage rounded to two decimal places
        overlap_pct = round(average_similarity * 100, 2)

        return overlap_pct

    # Get file data into dataframe
    def read_data_from_file(self, file_path):
        try:
            df = pd.read_excel(file_path)
            return df
        except FileNotFoundError:
            print(f"File Not Found: {file_path}.")
            return None

    def extract_data_from_file(self, file_path):
        df = self.read_data_from_file(file_path)
        if df is None:
            print("No data found in the specified file. Exiting.")
            return None

        # Define a function to process each row
        def process_row(row):
            ground_truth_entities = self.extract_named_entities(row['Ground_Truth'])
            output_entities = self.extract_named_entities(row['Output'])
            all_unique_entities = self.get_unique_entities(ground_truth_entities, output_entities)
            unique_in_output = self.get_unique_entities_in_output(output_entities, ground_truth_entities)
            unique_in_ground_truth = self.get_unique_entities_in_ground_truth(ground_truth_entities, output_entities)
            result = self.check_factual_accuracy(output_entities, ground_truth_entities)
            overlap_pct = self.calculate_overlap_pct(row['Ground_Truth'], row['Output'])

            return pd.Series({
                'Ground_Truth_Entities': ground_truth_entities,
                'Output_Entities': output_entities,
                'All_Unique_Entities': all_unique_entities,
                'Unique_In_Output': unique_in_output,
                'Unique_in_Ground_Truth': unique_in_ground_truth,
                'Result': result,
                'Overlap_PCT': overlap_pct
            })

        # Apply the processing function to each row
        result_df = df.apply(process_row, axis=1)

        # Concatenate the result with the original DataFrame
        df = pd.concat([df, result_df], axis=1)

        # Save the modified dataframe to a new CSV file
        df.to_excel(r"C:\Users\VishalChaurasiya(Ann\Downloads\L_D_Fluency_Validation_Results (2).xlsx", index=False)

        # Return the desired columns
        return df[
            ["Ground_Truth_Entities", "Output_Entities", "All_Unique_Entities", "Unique_In_Output",
             "Unique_in_Ground_Truth",
             "Result", "Similarity_PCT"]]

    def zast_comment(self):
        '''
        So, the above methods are used by the test cases.
        They should be implemented in your class if you want to run these tests.

        Thank You :) :)
        '''