import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
from pathlib import Path
from typing import List, Dict
from datetime import datetime


class Visualizer:
    """
    Creates all required visualizations
    """
    
    def __init__(self, output_dir: str = 'output'):
        """
        Initialize visualizer
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (14, 8)
    
    def create_all_visualizations(
        self,
        data: List[Dict],
        embeddings: np.ndarray,
        themes: List[Dict],
        temporal_data: Dict,
        anomalies: List[Dict],
        patterns: Dict
    ) -> Dict[str, str]:
        """
        Create all required visualizations
        
        Args:
            data: Processed entries
            embeddings: Embeddings array
            themes: List of themes
            temporal_data: Temporal analysis
            anomalies: Anomalies
            patterns: Patterns
            
        Returns:
            Dictionary mapping visualization names to file paths
        """
        viz_paths = {}
        
        print("  Creating 30-day timeline...")
        timeline_path = self._create_timeline(
            data, themes, temporal_data, anomalies
        )
        viz_paths['timeline'] = timeline_path
        
        print("  Creating weekly radar chart...")
        radar_path = self._create_weekly_radar(temporal_data, patterns)
        viz_paths['radar_chart'] = radar_path
        
        print("  Creating motif cluster graph...")
        print("  Creating motif cluster graph...")
        motif_path = self._create_motif_graph(themes, data)
        viz_paths['motif_graph'] = motif_path
        
        return viz_paths
    
    def _create_timeline(
        self,
        data: List[Dict],
        themes: List[Dict],
        temporal_data: Dict,
        anomalies: List[Dict]
    ) -> str:
        """
        Create 30-day multi-layer timeline
        
        Args:
            data: Processed entries
            themes: Themes
            temporal_data: Temporal data
            anomalies: Anomalies
            
        Returns:
            Path to saved visualization
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
        
        dates = [entry['date_obj'] for entry in data]
        date_strings = [entry['date'] for entry in data]
        
        theme_colors = plt.cm.Set3(np.linspace(0, 1, len(themes)))
        theme_map = {t['cluster_id']: (t['theme_label'], theme_colors[i]) 
                     for i, t in enumerate(themes)}
        
        for i, entry in enumerate(data):
            cluster = entry.get('cluster', 0)
            theme_label, color = theme_map.get(cluster, ('Unknown', 'gray'))
            ax1.bar(dates[i], 1, color=color, alpha=0.7, width=0.8)
        
        ax1.set_ylabel('Themes', fontsize=12, fontweight='bold')
        ax1.set_title('30-Day Multi-Layer Timeline: Themes, Mood, and Anomalies', 
                      fontsize=14, fontweight='bold', pad=20)
        ax1.set_ylim(0, 1.2)
        ax1.set_yticks([])
        
        handles = [plt.Rectangle((0,0),1,1, color=theme_map[t['cluster_id']][1], alpha=0.7) 
                   for t in themes]
        labels = [t['theme_label'] for t in themes]
        ax1.legend(handles, labels, loc='upper left', ncol=3, fontsize=9)
        
        sentiments = []
        for entry in data:
            text = entry.get('combined_text', '').lower()
            pos_words = ['good', 'great', 'happy', 'wonderful', 'amazing', 'love', 
                        'better', 'accomplished', 'grateful', 'fun', 'excited']
            neg_words = ['stress', 'pressure', 'anxious', 'nervous', 'worry', 
                        'tough', 'exhausted', 'tired', 'drained']
            pos = sum(1 for w in pos_words if w in text)
            neg = sum(1 for w in neg_words if w in text)
            sentiments.append(pos - neg)
        
        window = 3
        smoothed = np.convolve(sentiments, np.ones(window)/window, mode='same')
        
        ax2.plot(dates, smoothed, linewidth=2.5, color='#2E86AB', marker='o', 
                markersize=4, alpha=0.8)
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax2.fill_between(dates, smoothed, 0, where=(np.array(smoothed) >= 0), 
                         alpha=0.3, color='green', label='Positive')
        ax2.fill_between(dates, smoothed, 0, where=(np.array(smoothed) < 0), 
                         alpha=0.3, color='red', label='Negative')
        ax2.set_ylabel('Mood Score', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        anomaly_dates = [datetime.strptime(a['date'], '%Y-%m-%d') for a in anomalies]
        anomaly_types = [a['anomaly_type'] for a in anomalies]
        
        type_map = {
            'stress_surge': (0.8, '#E63946'),
            'fatigue_spike': (0.5, '#F77F00'),
            'confidence_dip': (0.2, '#FCA311'),
            'emotional_spike': (0.35, '#D62828')
        }
        
        for anom_date, anom_type in zip(anomaly_dates, anomaly_types):
            y_pos, color = type_map.get(anom_type, (0.5, 'red'))
            ax3.scatter(anom_date, y_pos, s=200, color=color, alpha=0.7, 
                       marker='X', edgecolors='black', linewidths=1.5)
            ax3.axvline(x=anom_date, color=color, alpha=0.2, linestyle='--', linewidth=1)
        
        ax3.set_ylabel('Anomalies', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax3.set_ylim(0, 1)
        ax3.set_yticks([])
        
        unique_types = list(set(anomaly_types))
        handles = [plt.scatter([], [], s=100, color=type_map[t][1], marker='X', 
                              edgecolors='black', linewidths=1) 
                  for t in unique_types if t in type_map]
        labels = [t.replace('_', ' ').title() for t in unique_types]
        if handles:
            ax3.legend(handles, labels, loc='upper left', ncol=2, fontsize=9)
        
        plt.xticks(rotation=45, ha='right')
        fig.autofmt_xdate()
        
        plt.tight_layout()
        
        output_path = str(self.output_dir / 'timeline.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _create_weekly_radar(
        self,
        temporal_data: Dict,
        patterns: Dict
    ) -> str:
        """
        Create weekly radar chart
        
        Args:
            temporal_data: Temporal data
            patterns: Pattern data
            
        Returns:
            Path to saved visualization
        """
        weekly_summaries = temporal_data['weekly_summaries']
        
        dimensions = ['Energy', 'Social', 'Work Focus', 'Wellness', 'Creativity']
        num_dims = len(dimensions)
        
        theme_scores = {
            'Work Performance': [0.7, 0.4, 0.9, 0.5, 0.6],
            'Social Connection': [0.6, 0.9, 0.5, 0.6, 0.5],
            'Rest & Recovery': [0.8, 0.5, 0.3, 0.9, 0.4],
            'Health & Wellness': [0.7, 0.5, 0.5, 0.9, 0.5],
            'Personal Growth': [0.6, 0.6, 0.7, 0.6, 0.8],
            'Leisure & Recreation': [0.8, 0.8, 0.4, 0.7, 0.7]
        }
        
        angles = np.linspace(0, 2 * np.pi, num_dims, endpoint=False).tolist()
        angles += angles[:1]  
        
        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(weekly_summaries)))
        
        for i, week in enumerate(weekly_summaries):
            theme = week['dominant_theme']
            scores = theme_scores.get(theme, [0.5] * num_dims).copy()
            
            if len(scores) > num_dims:
                scores = scores[:num_dims]
            elif len(scores) < num_dims:
                scores.extend([0.5] * (num_dims - len(scores)))
            
            if week['mood_trend'] == 'improving':
                scores = [min(1.0, s * 1.1) for s in scores]
            elif week['mood_trend'] == 'declining':
                scores = [s * 0.9 for s in scores]
            
            scores = scores + scores[:1]  
            ax.plot(angles, scores, 'o-', linewidth=2, label=f"Week {week['week']}", 
                   color=colors[i], alpha=0.7)
            ax.fill(angles, scores, alpha=0.15, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dimensions, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
        ax.set_title('Weekly Emotional Vectors Comparison', 
                     fontsize=14, fontweight='bold', pad=30)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
        
        plt.tight_layout()
        
        output_path = str(self.output_dir / 'radar_chart.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _create_motif_graph(
        self,
        themes: List[Dict],
        data: List[Dict]
    ) -> str:
        """
        Create motif cluster network graph
        
        Args:
            themes: List of themes
            data: Processed entries
            
        Returns:
            Path to saved visualization
        """
        G = nx.Graph()
        
        for theme in themes:
            G.add_node(
                theme['theme_label'],
                size=theme['entry_count'] * 100,
                cluster=theme['cluster_id']
            )
        
        for i in range(len(data) - 1):
            curr_cluster = data[i].get('cluster', 0)
            next_cluster = data[i + 1].get('cluster', 0)
            
            if curr_cluster != next_cluster:
                curr_theme = next((t['theme_label'] for t in themes 
                                 if t['cluster_id'] == curr_cluster), None)
                next_theme = next((t['theme_label'] for t in themes 
                                 if t['cluster_id'] == next_cluster), None)
                
                if curr_theme and next_theme:
                    if G.has_edge(curr_theme, next_theme):
                        G[curr_theme][next_theme]['weight'] += 1
                    else:
                        G.add_edge(curr_theme, next_theme, weight=1)
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        node_sizes = [G.nodes[node].get('size', 300) for node in G.nodes()]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(themes)))
        node_colors = [colors[G.nodes[node].get('cluster', 0)] for node in G.nodes()]
        
        nx.draw_networkx_nodes(
            G, pos, 
            node_size=node_sizes,
            node_color=node_colors,
            alpha=0.8,
            edgecolors='black',
            linewidths=2,
            ax=ax
        )
        
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        max_weight = max(weights) if weights else 1
        
        nx.draw_networkx_edges(
            G, pos,
            width=[3 * (w / max_weight) for w in weights],
            alpha=0.5,
            edge_color='gray',
            ax=ax
        )
        
        nx.draw_networkx_labels(
            G, pos,
            font_size=10,
            font_weight='bold',
            font_color='black',
            ax=ax
        )
        
        ax.set_title('Life Theme Relationship Network', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.axis('off')
        
        plt.tight_layout()
        
        output_path = str(self.output_dir / 'motif_graph.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
