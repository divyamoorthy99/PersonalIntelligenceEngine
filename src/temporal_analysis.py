import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
from collections import defaultdict


class TemporalAnalyzer:
    """
    Analyzes temporal evolution of themes and moods
    """
    
    def analyze(
        self, 
        embeddings: np.ndarray, 
        data: List[Dict], 
        themes: List[Dict]
    ) -> Dict:
        """
        Perform temporal analysis
        
        Args:
            embeddings: Embeddings array
            data: List of processed entries
            themes: List of theme dictionaries
            
        Returns:
            Temporal analysis results
        """
        weekly_data = self._group_by_week(data)
        
        weekly_summaries = []
        for week_num, week_entries in weekly_data.items():
            summary = self._analyze_week(
                week_num, week_entries, themes, embeddings, data
            )
            weekly_summaries.append(summary)
        
        return {
            'weekly_summaries': weekly_summaries,
            'total_weeks': len(weekly_summaries)
        }
    
    def _group_by_week(self, data: List[Dict]) -> Dict[int, List[Dict]]:
        """
        Group entries by week
        
        Args:
            data: List of entries
            
        Returns:
            Dictionary mapping week number to entries
        """
        weekly_data = defaultdict(list)
        
        start_date = data[0]['date_obj']
        
        for entry in data:
            days_diff = (entry['date_obj'] - start_date).days
            week_num = days_diff // 7 + 1
            entry['week'] = week_num
            weekly_data[week_num].append(entry)
        
        return weekly_data
    
    def _analyze_week(
        self,
        week_num: int,
        week_entries: List[Dict],
        themes: List[Dict],
        embeddings: np.ndarray,
        all_data: List[Dict]
    ) -> Dict:
        """
        Analyze a single week
        
        Args:
            week_num: Week number
            week_entries: Entries in this week
            themes: List of themes
            embeddings: All embeddings
            all_data: All entries
            
        Returns:
            Weekly summary dictionary
        """
        if week_entries:
            cluster_counts = {}
            for entry in week_entries:
                cluster = entry.get('cluster', 0)
                cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
            
            dominant_cluster = max(cluster_counts, key=cluster_counts.get)
            dominant_theme = next(
                (t for t in themes if t['cluster_id'] == dominant_cluster),
                themes[0]
            )
        else:
            dominant_theme = themes[0]
        
        mood_trend = self._calculate_mood_trend(week_entries, all_data, embeddings)
        
        week_dates = [entry['date'] for entry in week_entries]
        
        micro_insight = self._generate_micro_insight(
            week_entries, dominant_theme, mood_trend
        )
        
        return {
            'week': week_num,
            'start_date': week_entries[0]['date'] if week_entries else None,
            'end_date': week_entries[-1]['date'] if week_entries else None,
            'dominant_theme': dominant_theme['theme_label'],
            'mood_trend': mood_trend,
            'entry_count': len(week_entries),
            'micro_insight': micro_insight
        }
    
    def _calculate_mood_trend(
        self, 
        week_entries: List[Dict],
        all_data: List[Dict],
        embeddings: np.ndarray
    ) -> str:
        """
        Calculate mood trend for the week
        
        Args:
            week_entries: Entries in the week
            all_data: All entries
            embeddings: All embeddings
            
        Returns:
            Mood trend label
        """
        if len(week_entries) < 2:
            return "stable"
        
        sentiments = []
        for entry in week_entries:
            text = entry.get('combined_text', '').lower()
            
            positive_words = [
                'good', 'great', 'happy', 'wonderful', 'amazing', 'love',
                'better', 'accomplished', 'grateful', 'fun', 'excited',
                'relieved', 'positive', 'motivated', 'confident', 'inspired'
            ]
            
            negative_words = [
                'stress', 'pressure', 'anxious', 'nervous', 'worry', 'tough',
                'exhausted', 'tired', 'drained', 'difficult', 'hard', 'sick',
                'worried', 'unprepared', 'uncertain'
            ]
            
            pos_count = sum(1 for word in positive_words if word in text)
            neg_count = sum(1 for word in negative_words if word in text)
            
            sentiment = pos_count - neg_count
            sentiments.append(sentiment)
        
        if len(sentiments) >= 2:
            first_half_avg = np.mean(sentiments[:len(sentiments)//2])
            second_half_avg = np.mean(sentiments[len(sentiments)//2:])
            
            diff = second_half_avg - first_half_avg
            
            if diff > 0.5:
                return "improving"
            elif diff < -0.5:
                return "declining"
            else:
                return "stable"
        
        return "stable"
    
    def _generate_micro_insight(
        self,
        week_entries: List[Dict],
        dominant_theme: Dict,
        mood_trend: str
    ) -> str:
        """
        Generate micro-insight for the week
        
        Args:
            week_entries: Entries in the week
            dominant_theme: Dominant theme
            mood_trend: Mood trend
            
        Returns:
            Micro-insight text
        """
        theme_label = dominant_theme['theme_label']
        
        insights = {
            'improving': f"{theme_label} shows positive progression. Consider maintaining current strategies.",
            'declining': f"{theme_label} indicates increasing challenges. Consider seeking support or adjusting approach.",
            'stable': f"{theme_label} remains consistent. Current balance appears sustainable."
        }
        
        return insights.get(mood_trend, f"{theme_label} is the focus this week.")
