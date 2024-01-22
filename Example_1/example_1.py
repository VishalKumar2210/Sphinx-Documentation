import spacy
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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
        named_entities = [ent.text for ent in doc.ents]
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
        if not ground_truth_entities:
            return False

        for fact in ground_truth_entities:
            if not any(fact in entity for entity in output_entities):
                return False

        return True

    def unique_entities(self, text: str) -> set:
        """
        Return a unique set of found entities in the given text.
        Parameters:
        - text (str): The input text.
        Returns:
        - set: Unique set of named entities.
        """
        doc = self.nlp(text)
        named_entities = set(ent.text for ent in doc.ents)
        return named_entities

    def get_unique_entities(self, output_entities: list, ground_truth_entities: list) -> list:
        """
        Return a unique list of entities found in both output_entities and ground_truth_entities.
        Parameters:
        - output_entities (list): List of named entities from the output.
        - ground_truth_entities (list): List of ground truth named entities.
        Returns:
        - list: Unique list of found entities.
        """
        unique_entities = set(output_entities) | set(ground_truth_entities)
        return list(unique_entities)

    def get_unique_entities_in_output(self, output_entities: list, ground_truth_entities: list) -> set:
        """
        Return the unique set of entities found in output_entities but not in ground_truth_entities.
        Parameters:
        - output_entities (list): List of named entities from the output.
        - ground_truth_entities (list): List of ground truth named entities.
        Returns:
        - set: Unique set of entities found in output_entities but not in ground_truth_entities.
        """
        unique_set_in_output = set(output_entities) - set(ground_truth_entities)
        return unique_set_in_output

    def get_unique_entities_in_ground_truth(self, ground_truth_entities: list, output_entities: list) -> set:
        """
        Return the unique set of entities found in ground_truth_entities but not in output_entities.
        Parameters:
        - ground_truth_entities (list): List of ground truth named entities.
        - output_entities (list): List of named entities from the output.
        Returns:
        - set: Unique set of entities found in ground_truth_entities but not in output_entities.
        """
        unique_set_in_ground_truth = set(ground_truth_entities) - set(output_entities)
        return unique_set_in_ground_truth

    def calculate_cosine_similarity(self, output_entities: list, ground_truth_entities: list) -> float:
        """
        Calculate the cosine similarity between output_entities and ground_truth_entities.
        Parameters:
        - output_entities (list): List of named entities from the output.
        - ground_truth_entities (list): List of ground truth named entities.
        Returns:
        - float: Cosine similarity between the two sets of entities as a percentage.
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
        similarity_percentage = round(average_similarity * 100, 2)

        return similarity_percentage
    

    def zast_comment(self):
        '''
        So, the above methods are used by the test cases.
        They should be implemented in your class if you want to run these tests.

        Thank You :) :)
        '''
        
    
    