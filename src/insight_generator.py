from typing import List, Dict
import numpy as np


class InsightGenerator:
    """
    Generates micro, macro, and predictive insights
    """
    
    def generate(
        self,
        themes: List[Dict],
        temporal_data: Dict,
        anomalies: List[Dict],
        patterns: Dict,
        data: List[Dict]
    ) -> Dict:
        """
        Generate all types of insights
        
        Args:
            themes: List of themes
            temporal_data: Temporal analysis results
            anomalies: List of anomalies
            patterns: Pattern analysis results
            data: Processed entries
            
        Returns:
            Dictionary containing all insights
        """
        macro_insight = self._generate_macro_insight(
            themes, temporal_data, anomalies, patterns
        )
        
        predictive_insight = self._generate_predictive_insight(
            temporal_data, patterns, data
        )
        
        safety_notes = self._generate_safety_notes(anomalies, data)
        
        return {
            'macro_insight': macro_insight,
            'predictive_insight': predictive_insight,
            'safety_notes': safety_notes
        }
    
    def _generate_macro_insight(
        self,
        themes: List[Dict],
        temporal_data: Dict,
        anomalies: List[Dict],
        patterns: Dict
    ) -> str:
        """
        Generate macro insight for entire period
        
        Args:
            themes: List of themes
            temporal_data: Temporal data
            anomalies: Anomalies
            patterns: Patterns
            
        Returns:
            Macro insight text
        """
        dominant_theme = max(themes, key=lambda t: t['entry_count'])
        
        weekly_summaries = temporal_data['weekly_summaries']
        trend_counts = {}
        for week in weekly_summaries:
            trend = week['mood_trend']
            trend_counts[trend] = trend_counts.get(trend, 0) + 1
        
        dominant_trend = max(trend_counts, key=trend_counts.get) if trend_counts else 'stable'
        
        insight_parts = [
            f"Over the 30-day period, life patterns were primarily characterized by {dominant_theme['theme_label'].lower()}.",
        ]
        
        if dominant_trend == 'improving':
            insight_parts.append("Overall emotional trajectory shows positive growth.")
        elif dominant_trend == 'declining':
            insight_parts.append("Some challenging periods were observed, suggesting need for additional support strategies.")
        else:
            insight_parts.append("Emotional patterns remained relatively balanced throughout the period.")
        
        if patterns['weekly_cycle_detected']:
            insight_parts.append(patterns['description'])
        
        if len(anomalies) > 0:
            insight_parts.append(f"{len(anomalies)} significant emotional events were identified.")
        
        return " ".join(insight_parts)
    
    def _generate_predictive_insight(
        self,
        temporal_data: Dict,
        patterns: Dict,
        data: List[Dict]
    ) -> str:
        """
        Generate predictive insight
        
        Args:
            temporal_data: Temporal data
            patterns: Patterns
            data: Processed entries
            
        Returns:
            Predictive insight text
        """
        weekly_summaries = temporal_data['weekly_summaries']
        
        if len(weekly_summaries) < 2:
            return "Insufficient data for reliable prediction."
        
        recent_weeks = weekly_summaries[-2:]
        
        recent_trends = [w['mood_trend'] for w in recent_weeks]
        recent_themes = [w['dominant_theme'] for w in recent_weeks]
        
        prediction_parts = []
        
        if all(t == 'declining' for t in recent_trends):
            prediction_parts.append("If current pattern continues, Week 5 may show continued challenges.")
            prediction_parts.append("Consider implementing stress-reduction strategies proactively.")
        elif all(t == 'improving' for t in recent_trends):
            prediction_parts.append("If momentum continues, Week 5 likely to show sustained positive trajectory.")
            prediction_parts.append("Consider strategies to maintain current positive practices.")
        else:
            prediction_parts.append("Week 5 may show similar patterns to recent weeks with typical weekly fluctuations.")
        
        if len(set(recent_themes)) == 1:
            theme = recent_themes[0]
            prediction_parts.append(f"{theme} is likely to remain a central focus.")
        
        return " ".join(prediction_parts)
    
    def _generate_safety_notes(
        self,
        anomalies: List[Dict],
        data: List[Dict]
    ) -> List[str]:
        """
        Generate safety notes
        
        Args:
            anomalies: List of anomalies
            data: Processed entries
            
        Returns:
            List of safety notes
        """
        notes = []
        
        risk_words = ['hopeless', 'worthless', 'give up', 'can\'t go on', 'suicide', 'self-harm']
        high_risk_detected = False
        
        for entry in data:
            text = entry.get('combined_text', '').lower()
            if any(word in text for word in risk_words):
                high_risk_detected = True
                break
        
        if high_risk_detected:
            notes.append("High-risk emotional indicators detected. Professional consultation strongly recommended.")
        else:
            notes.append("No critical risk indicators detected.")
        
        ambiguous_words = ['uncertain', 'doubt', 'worried', 'unprepared']
        ambiguous_count = 0
        ambiguous_dates = []
        
        for entry in data:
            text = entry.get('combined_text', '').lower()
            if any(word in text for word in ambiguous_words):
                ambiguous_count += 1
                ambiguous_dates.append(entry['date'])
        
        if ambiguous_count > 0:
            notes.append(
                f"Ambiguous self-doubt language detected on {ambiguous_count} occasions. "
                "Interpretations should consider broader context."
            )
        
        notes.append(
            "All insights are observational and non-diagnostic. "
            "This analysis does not replace professional mental health assessment."
        )
        
        return notes
