import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from typing import List, Dict
import re


class ThemeClusterer:
    """
    Clusters entries into life themes
    """
    
    def __init__(self, n_clusters: int = 5):
        """
        Initialize clusterer
        
        Args:
            n_clusters: Number of themes to identify
        """
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        
    def fit_predict(self, embeddings: np.ndarray, data: List[Dict]) -> List[Dict]:
        """
        Cluster embeddings and extract themes
        
        Args:
            embeddings: Array of embeddings
            data: List of processed entries
            
        Returns:
            List of theme dictionaries
        """
        labels = self.model.fit_predict(embeddings)
        
        for i, entry in enumerate(data):
            entry['cluster'] = int(labels[i])
        
        themes = []
        for cluster_id in range(self.n_clusters):
            theme = self._extract_theme(cluster_id, data, embeddings, labels)
            themes.append(theme)
        
        return themes
    
    def _extract_theme(
        self, 
        cluster_id: int, 
        data: List[Dict], 
        embeddings: np.ndarray, 
        labels: np.ndarray
    ) -> Dict:
        """
        Extract theme information for a cluster
        
        Args:
            cluster_id: Cluster ID
            data: List of entries
            embeddings: Embeddings array
            labels: Cluster labels
            
        Returns:
            Theme dictionary
        """
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_entries = [data[i] for i in cluster_indices]
        
        centroid = self.model.cluster_centers_[cluster_id]
        distances = [
            np.linalg.norm(embeddings[i] - centroid) 
            for i in cluster_indices
        ]
        sorted_indices = sorted(
            range(len(distances)), 
            key=lambda i: distances[i]
        )[:3]
        representative_entry_ids = [
            cluster_entries[i]['entry_id'] 
            for i in sorted_indices
        ]
        
        keywords = self._extract_keywords(cluster_entries)
        
        theme_label = self._generate_theme_label(keywords, cluster_entries)
        
        confidence = self._calculate_confidence(
            embeddings, labels, cluster_id, centroid
        )
        
        return {
            'theme_label': theme_label,
            'cluster_id': cluster_id,
            'representative_entries': representative_entry_ids,
            'keywords': keywords[:10],
            'cluster_confidence': round(confidence, 2),
            'entry_count': len(cluster_entries)
        }
    
    def _extract_keywords(self, entries: List[Dict]) -> List[str]:
        """
        Extract keywords from cluster entries
        
        Args:
            entries: List of entries in cluster
            
        Returns:
            List of keywords
        """
        all_text = " ".join([
            entry.get('combined_text', '') 
            for entry in entries
        ]).lower()
        
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
            'for', 'of', 'with', 'by', 'from', 'up', 'about', 'is', 'was',
            'are', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can',
            'diary', 'voice', 'scene', 'this', 'that', 'i', 'my', 'me'
        }
        
        words = re.findall(r'\b[a-z]{4,}\b', all_text)
        words = [w for w in words if w not in stop_words]
        
        word_counts = Counter(words)
        
        return [word for word, count in word_counts.most_common(15)]
    
    def _generate_theme_label(
        self, 
        keywords: List[str], 
        entries: List[Dict]
    ) -> str:
        """
        Generate a theme label based on keywords and entries
        
        Args:
            keywords: List of keywords
            entries: List of entries
            
        Returns:
            Theme label
        """
        theme_keywords = {
            'Work Performance': ['work', 'project', 'deadline', 'meeting', 'team', 'review', 'presentation', 'client', 'office'],
            'Social Connection': ['friends', 'family', 'conversation', 'together', 'people', 'colleague', 'bonding'],
            'Rest & Recovery': ['weekend', 'relax', 'rest', 'sleep', 'tired', 'recharged', 'break', 'vacation'],
            'Health & Wellness': ['exercise', 'health', 'sick', 'recover', 'energy', 'running', 'wellness'],
            'Personal Growth': ['learning', 'mentor', 'creative', 'goal', 'reflection', 'journey', 'growth'],
            'Leisure & Recreation': ['beach', 'hiking', 'music', 'concert', 'movie', 'fun', 'entertainment']
        }
        
        theme_scores = {}
        for theme, theme_words in theme_keywords.items():
            score = sum(1 for kw in keywords if kw in theme_words)
            theme_scores[theme] = score
        
        if max(theme_scores.values()) > 0:
            return max(theme_scores, key=theme_scores.get)
        
        return " ".join(keywords[:2]).title()
    
    def _calculate_confidence(
        self, 
        embeddings: np.ndarray, 
        labels: np.ndarray, 
        cluster_id: int,
        centroid: np.ndarray
    ) -> float:
        """
        Calculate cluster confidence score
        
        Args:
            embeddings: Embeddings array
            labels: Cluster labels
            cluster_id: Cluster ID
            centroid: Cluster centroid
            
        Returns:
            Confidence score (0-1)
        """
        cluster_indices = np.where(labels == cluster_id)[0]
        
        if len(cluster_indices) == 0:
            return 0.0
        
        distances = [
            np.linalg.norm(embeddings[i] - centroid)
            for i in cluster_indices
        ]
        avg_distance = np.mean(distances)
        
        confidence = np.exp(-avg_distance / 2)
        
        return float(confidence)
