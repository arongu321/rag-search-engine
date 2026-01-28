import json
import pickle
import os

class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}
    
    def __add_document(self, doc_id, text):
        """Add a document to the inverted index."""
        self.docmap[doc_id] = text
        words = text.split()
        for word in words:
            if word.lower() not in self.index:
                self.index[word.lower()] = set()
            self.index[word.lower()].add(doc_id)
    
    def get_documents(self, term):
        """Retrieve documents containing the given term in ascending order of document IDs."""
        if term.lower() in self.index:
            return sorted(self.index[term.lower()])
        else:
            return []
    
    def build(self):
        """Build inverted index from documents in data/movies.json."""
        with open("data/movies.json", "r") as f:
            data = json.load(f)
            movies = data["movies"]
            for movie in movies:
                self.__add_document(movie['id'], f"{movie['title']} {movie['description']}")
    
    def save(self):
        """Save the inverted index and document map to disk."""
        if not os.path.exists("cache"):
            os.makedirs("cache")
        with open("cache/index.pkl", "wb") as f:
            pickle.dump(self.index, f)
        with open("cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)
        