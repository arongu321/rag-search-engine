from PIL import Image
from sentence_transformers import SentenceTransformer
import numpy as np
from lib.search_utils import (
    load_movies,
)

def cosine_similarity_numpy(vec1, vec2):
  """
  Calculates the cosine similarity between two 1D NumPy vectors.
  
  Args:
    vec1 (np.ndarray): The first vector.
    vec2 (np.ndarray): The second vector.
    
  Returns:
    float: The cosine similarity score between -1 and 1.
  """
  dot_product = np.dot(vec1, vec2)
  norm_vec1 = np.linalg.norm(vec1)
  norm_vec2 = np.linalg.norm(vec2)
  
  if norm_vec1 == 0 or norm_vec2 == 0:
    return 0  # Or handle as appropriate for your use case (e.g., raise an error)

  similarity = dot_product / (norm_vec1 * norm_vec2)
  return similarity

class MultiModalSearch:
    def __init__(self, model_name="clip-ViT-B-32", documents=None):
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.texts = [f"{doc['title']}: {doc['description']}" for doc in documents] if documents else []
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True) if documents else None
        
    def embed_image(self, image_path):
        image = Image.open(image_path)
        embedding = self.model.encode([image])
        return embedding[0]

    def search_with_image(self, image_path):
        image_embedding = self.embed_image(image_path)
        if self.text_embeddings is None:
            raise ValueError("No document embeddings available for search.")

        similarities = np.array([cosine_similarity_numpy(image_embedding, text_embedding) for text_embedding in self.text_embeddings])
        
        # Take first 5 results
        ranked_indices = np.argsort(similarities)[::-1][:5]
        
        results = []
        for idx in ranked_indices:
            results.append({
                "id": self.documents[idx]['id'],
                "title": self.documents[idx]['title'],
                "description": self.documents[idx]['description'],
                "score": similarities[idx]
            })
        return results

def verify_image_embedding(image_path):
    search = MultiModalSearch()
    embedding = search.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")

def image_search_command(image_path):
    movies = load_movies()
    multimodal_search = MultiModalSearch(documents=movies)
    results = multimodal_search.search_with_image(image_path)
    return results
    