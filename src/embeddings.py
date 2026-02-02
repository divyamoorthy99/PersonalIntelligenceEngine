import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple


class MultiModalEmbedder:
    """
    Generates multi-modal embeddings from text, voice, and image data
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize embedder with sentence transformer model
        
        Args:
            model_name: Name of the sentence transformer model
        """
        print(f"  Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
    def generate_embeddings(self, data: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
        """
        Generate multi-modal embeddings for all entries
        
        Args:
            data: List of entry dictionaries
            
        Returns:
            Tuple of (embeddings array, processed data)
        """
        embeddings_list = []
        processed_data = []
        
        for entry in data:
            combined_text = self._combine_modalities(entry)
            
            embedding = self.model.encode(combined_text)
            embeddings_list.append(embedding)
            
            processed_entry = entry.copy()
            processed_entry['combined_text'] = combined_text
            processed_entry['embedding'] = embedding
            processed_data.append(processed_entry)
        
        embeddings = np.array(embeddings_list)
        return embeddings, processed_data
    
    def _combine_modalities(self, entry: Dict) -> str:
        """
        Combine text, voice transcript, and image caption into single text
        
        Args:
            entry: Entry dictionary
            
        Returns:
            Combined text string
        """
        parts = []
        
        if entry.get('text'):
            parts.append(f"Diary: {entry['text']}")
        
        if entry.get('voice_transcript'):
            parts.append(f"Voice: {entry['voice_transcript']}")
        
        if entry.get('image_caption'):
            parts.append(f"Scene: {entry['image_caption']}")
        
        return " ".join(parts)
