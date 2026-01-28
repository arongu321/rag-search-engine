import os
import pickle
import string
import math
from collections import defaultdict, Counter
from nltk.stem import PorterStemmer
from .search_utils import (
    BM25_K1,
    BM25_B,
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    INVERTED_INDEX_PATH,
    TERM_FREQUENCIES_PATH,
    DOC_LENGTHS_PATH,
    DOC_MAP_PATH,
    load_movies,
    load_stopwords,
    format_search_result,
)

class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.term_frequencies = defaultdict(Counter)
        self.doc_lengths = {}
        self.index_path = INVERTED_INDEX_PATH
        self.docmap_path = DOC_MAP_PATH
        self.term_frequencies_path = TERM_FREQUENCIES_PATH
        self.doc_lengths_path = DOC_LENGTHS_PATH

    def build(self) -> None:
        movies = load_movies()
        for m in movies:
            doc_id = m["id"]
            doc_description = f"{m['title']} {m['description']}"
            self.docmap[doc_id] = m
            self.__add_document(doc_id, doc_description)

    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self) -> None:
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.term_frequencies_path, "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += tokens.count(token)
        
        # Count number of tokens in each document
        for doc_id in self.term_frequencies:
            self.doc_lengths[doc_id] = sum(self.term_frequencies[doc_id].values())
    
    def __get_avg_doc_length(self) -> float:
        total_length = sum(self.doc_lengths.values())
        return total_length / len(self.doc_lengths) if self.doc_lengths else 0.0
    
    def get_tf(self, doc_id: int, term: str) -> int:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        return self.term_frequencies[doc_id][token] if doc_id in self.term_frequencies else 0
    
    def get_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.index[token]) if token in self.index else 0
        idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1)) 
        return idf

    def get_tf_idf(self, doc_id: int, term: str) -> float:
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf

    def get_bm25_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.index[token]) if token in self.index else 0
        idf = math.log((total_doc_count - term_match_doc_count + 0.5) / (term_match_doc_count + 0.5) + 1)
        return idf
    
    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B) -> float:
        tf = self.get_tf(doc_id, term)
        avg_doc_length = self.__get_avg_doc_length()
        doc_length = self.doc_lengths.get(doc_id, 0)
        bm25_tf = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_length / avg_doc_length)))
        return bm25_tf
    
    def bm25(self, doc_id, term) -> float:
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_tf * bm25_idf
    
    def bm25_search(self, query, limit):
        tokens = tokenize_text(query)
        scores = defaultdict(float)
        for document in self.docmap.values():
            doc_id = document["id"]
            for token in tokens:
                scores[doc_id] += self.bm25(doc_id, token)
        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]
        
        results = []
        for doc_id, score in ranked_docs:
            doc = self.docmap[doc_id]
            formatted_res = format_search_result(
                doc_id=doc_id,
                title=doc["title"],
                document=doc["description"][:100],
                score=score,
            )
            results.append(formatted_res)
        return results

def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    idx.load()
    query_tokens = tokenize_text(query)
    seen, results = set(), []
    for query_token in query_tokens:
        matching_doc_ids = idx.get_documents(query_token)
        for doc_id in matching_doc_ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            doc = idx.docmap[doc_id]
            results.append(doc)
            if len(results) >= limit:
                return results

    return results

def tf_command(doc_id: int, term: str) -> int:
    idx = InvertedIndex()
    idx.load()
    processed_term = preprocess_text(term)
    return idx.get_tf(doc_id, processed_term)

def idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    processed_term = preprocess_text(term)
    idf = idx.get_idf(processed_term)
    return idf

def tf_idf_command(doc_id: int, term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    processed_term = preprocess_text(term)
    return idx.get_tf_idf(doc_id, processed_term)

def bm25_idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    processed_term = preprocess_text(term)
    return idx.get_bm25_idf(processed_term)

def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1) -> float:
    idx = InvertedIndex()
    idx.load()
    processed_term = preprocess_text(term)
    return idx.get_bm25_tf(doc_id, processed_term, k1)

def bm25_search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    idx.load()
    ranked_docs = idx.bm25_search(query, limit)
    results = []
    for doc_id, score in ranked_docs:
        doc = idx.docmap[doc_id]
        results.append((doc, score))
    return results

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)
    stop_words = load_stopwords()
    filtered_words = []
    for word in valid_tokens:
        if word not in stop_words:
            filtered_words.append(word)
    stemmer = PorterStemmer()
    stemmed_words = []
    for word in filtered_words:
        stemmed_words.append(stemmer.stem(word))
    return stemmed_words