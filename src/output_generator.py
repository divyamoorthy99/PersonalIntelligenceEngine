from typing import Dict, List


class OutputGenerator:
    """
    Generates final JSON output
    """
    
    def generate_output(
        self,
        themes: List[Dict],
        temporal_data: Dict,
        anomalies: List[Dict],
        patterns: Dict,
        insights: Dict,
        viz_paths: Dict[str, str]
    ) -> Dict:
        """
        Generate final output JSON
        
        Args:
            themes: List of themes
            temporal_data: Temporal analysis
            anomalies: Anomalies
            patterns: Patterns
            insights: Generated insights
            viz_paths: Visualization file paths
            
        Returns:
            Complete output dictionary
        """
        output = {
            'themes': themes,
            'temporal_evolution': temporal_data['weekly_summaries'],
            'pattern_cycles': {
                'weekly_cycle_detected': patterns['weekly_cycle_detected'],
                'description': patterns['description'],
                'day_of_week_patterns': patterns['day_of_week_patterns']
            },
            'anomalies': anomalies,
            'macro_insight': insights['macro_insight'],
            'predictive_insight': insights['predictive_insight'],
            'safety_notes': insights['safety_notes'],
            'visualizations': {
                'timelines_visualization': viz_paths.get('timeline', ''),
                'weekly_radar_chart': viz_paths.get('radar_chart', ''),
                'motifs_graph': viz_paths.get('motif_graph', '')
            }
        }
        
        return output
