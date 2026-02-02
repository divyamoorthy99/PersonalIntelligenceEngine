import numpy as np
from sklearn.ensemble import IsolationForest
from typing import List, Dict


class AnomalyDetector:
    """
    Detects anomalies in emotional patterns
    """
    
    def __init__(self, contamination: float = 0.1):
        """
        Initialize anomaly detector
        
        Args:
            contamination: Expected proportion of anomalies
        """
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42
        )
    
    def detect(
        self, 
        embeddings: np.ndarray, 
        data: List[Dict]
    ) -> List[Dict]:
        """
        Detect anomalies in the data
        
        Args:
            embeddings: Embeddings array
            data: List of processed entries
            
        Returns:
            List of anomaly dictionaries
        """
        predictions = self.model.fit_predict(embeddings)
        
        scores = self.model.score_samples(embeddings)
        
        anomalies = []
        for i, (pred, score) in enumerate(zip(predictions, scores)):
            if pred == -1:  
                anomaly = self._create_anomaly_entry(
                    data[i], score, embeddings, i
                )
                anomalies.append(anomaly)
        
        anomalies.sort(key=lambda x: x['anomaly_score'])
        
        return anomalies
    
    def _create_anomaly_entry(
        self,
        entry: Dict,
        score: float,
        embeddings: np.ndarray,
        index: int
    ) -> Dict:
        """
        Create anomaly entry with details
        
        Args:
            entry: Data entry
            score: Anomaly score
            embeddings: All embeddings
            index: Entry index
            
        Returns:
            Anomaly dictionary
        """
        text = entry.get('combined_text', '').lower()
        
        anomaly_type = "emotional_spike"
        if any(word in text for word in ['stress', 'pressure', 'anxious', 'nervous']):
            anomaly_type = "stress_surge"
        elif any(word in text for word in ['sick', 'tired', 'exhausted', 'drained']):
            anomaly_type = "fatigue_spike"
        elif any(word in text for word in ['unprepared', 'worry', 'uncertain', 'doubt']):
            anomaly_type = "confidence_dip"
        
        return {
            'entry_id': entry['entry_id'],
            'date': entry['date'],
            'anomaly_type': anomaly_type,
            'anomaly_score': round(float(score), 4),
            'description': self._generate_anomaly_description(entry, anomaly_type)
        }
    
    def _generate_anomaly_description(
        self, 
        entry: Dict, 
        anomaly_type: str
    ) -> str:
        """
        Generate description for anomaly
        
        Args:
            entry: Data entry
            anomaly_type: Type of anomaly
            
        Returns:
            Description text
        """
        descriptions = {
            'stress_surge': f"Elevated stress levels detected on {entry['date']}",
            'fatigue_spike': f"Significant fatigue indicators on {entry['date']}",
            'confidence_dip': f"Confidence or self-doubt concerns on {entry['date']}",
            'emotional_spike': f"Unusual emotional pattern detected on {entry['date']}"
        }
        
        return descriptions.get(
            anomaly_type, 
            f"Anomaly detected on {entry['date']}"
        )
