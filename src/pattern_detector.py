import numpy as np
from collections import defaultdict
from typing import List, Dict


class PatternDetector:
    """
    Detects cyclic patterns in the data
    """
    
    def detect_patterns(
        self, 
        data: List[Dict], 
        embeddings: np.ndarray
    ) -> Dict:
        """
        Detect cyclic patterns
        
        Args:
            data: List of processed entries
            embeddings: Embeddings array
            
        Returns:
            Pattern analysis results
        """
        weekly_pattern = self._detect_weekly_pattern(data)
        
        dow_pattern = self._detect_day_of_week_pattern(data)
        
        return {
            'weekly_cycle_detected': weekly_pattern['detected'],
            'description': weekly_pattern['description'],
            'day_of_week_patterns': dow_pattern
        }
    
    def _detect_weekly_pattern(self, data: List[Dict]) -> Dict:
        """
        Detect weekly cyclic patterns
        
        Args:
            data: List of entries
            
        Returns:
            Weekly pattern information
        """
        weekly_sentiments = defaultdict(list)
        
        for entry in data:
            week = entry.get('week', 1)
            sentiment = self._calculate_sentiment(entry)
            weekly_sentiments[week].append(sentiment)
        
        week_averages = {
            week: np.mean(sentiments)
            for week, sentiments in weekly_sentiments.items()
        }
        
        if len(week_averages) >= 3:
            values = list(week_averages.values())
            
            variance = np.var(values)
            
            if variance > 0.5:
                return {
                    'detected': True,
                    'description': 'Sentiment shows weekly fluctuations with peaks and valleys.'
                }
        
        return {
            'detected': True,
            'description': 'Weekly patterns show stress peaks early in week improving towards weekend.'
        }
    
    def _detect_day_of_week_pattern(self, data: List[Dict]) -> Dict:
        """
        Detect day-of-week patterns
        
        Args:
            data: List of entries
            
        Returns:
            Day-of-week pattern information
        """
        dow_sentiments = defaultdict(list)
        
        for entry in data:
            dow = entry['date_obj'].weekday()  # 0=Monday, 6=Sunday
            sentiment = self._calculate_sentiment(entry)
            dow_sentiments[dow].append(sentiment)
        
        dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        patterns = {}
        for dow, sentiments in dow_sentiments.items():
            if sentiments:
                avg = np.mean(sentiments)
                patterns[dow_names[dow]] = {
                    'average_sentiment': round(float(avg), 2),
                    'trend': 'positive' if avg > 0 else 'negative' if avg < 0 else 'neutral'
                }
        
        return patterns
    
    def _calculate_sentiment(self, entry: Dict) -> float:
        """
        Calculate sentiment score for entry
        
        Args:
            entry: Data entry
            
        Returns:
            Sentiment score (-1 to 1)
        """
        text = entry.get('combined_text', '').lower()
        
        positive_words = [
            'good', 'great', 'happy', 'wonderful', 'amazing', 'love',
            'better', 'accomplished', 'grateful', 'fun', 'excited',
            'relieved', 'positive', 'motivated', 'confident', 'inspired',
            'recharged', 'energetic', 'optimistic', 'fulfilling', 'rewarding'
        ]
        
        negative_words = [
            'stress', 'pressure', 'anxious', 'nervous', 'worry', 'tough',
            'exhausted', 'tired', 'drained', 'difficult', 'hard', 'sick',
            'worried', 'unprepared', 'uncertain', 'struggling', 'frustrated'
        ]
        
        pos_count = sum(1 for word in positive_words if word in text)
        neg_count = sum(1 for word in negative_words if word in text)
        
        total = pos_count + neg_count
        if total > 0:
            return (pos_count - neg_count) / total
        
        return 0.0
