import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_data
from src.embeddings import MultiModalEmbedder
from src.clustering import ThemeClusterer
from src.temporal_analysis import TemporalAnalyzer
from src.anomaly_detector import AnomalyDetector
from src.pattern_detector import PatternDetector
from src.insight_generator import InsightGenerator
from src.visualizer import Visualizer
from src.output_generator import OutputGenerator


def main():
    """
    Main pipeline execution
    """
    print("=" * 60)
    print("Multi-Modal Life Pattern Intelligence Pipeline")
    print("=" * 60)
    
    # Load data
    print("\n[1/9] Loading data...")
    data = load_data('data/input_data.json')
    print(f"✓ Loaded {len(data)} entries")
    
    # Generate multi-modal embeddings
    print("\n[2/9] Generating multi-modal embeddings...")
    embedder = MultiModalEmbedder()
    embeddings, processed_data = embedder.generate_embeddings(data)
    print(f"✓ Generated embeddings of shape {embeddings.shape}")
    
    # Cluster into themes
    print("\n[3/9] Clustering into life themes...")
    clusterer = ThemeClusterer(n_clusters=5)
    themes = clusterer.fit_predict(embeddings, processed_data)
    print(f"✓ Identified {len(themes)} themes")
    
    # Temporal evolution analysis
    print("\n[4/9] Analyzing temporal evolution...")
    temporal = TemporalAnalyzer()
    temporal_data = temporal.analyze(embeddings, processed_data, themes)
    print(f"✓ Analyzed {len(temporal_data['weekly_summaries'])} weeks")
    
    # Detect anomalies
    print("\n[5/9] Detecting anomalies...")
    anomaly_det = AnomalyDetector()
    anomalies = anomaly_det.detect(embeddings, processed_data)
    print(f"✓ Detected {len(anomalies)} anomalies")
    
    # Detect cyclic patterns
    print("\n[6/9] Detecting cyclic patterns...")
    pattern_det = PatternDetector()
    patterns = pattern_det.detect_patterns(processed_data, embeddings)
    print(f"✓ Detected cyclic patterns")
    
    # Generate insights
    print("\n[7/9] Generating insights...")
    insight_gen = InsightGenerator()
    insights = insight_gen.generate(
        themes, temporal_data, anomalies, patterns, processed_data
    )
    print("✓ Generated micro, macro, and predictive insights")
    
    # Create visualizations
    print("\n[8/9] Creating visualizations...")
    visualizer = Visualizer()
    viz_paths = visualizer.create_all_visualizations(
        processed_data, embeddings, themes, temporal_data, anomalies, patterns
    )
    print(f"✓ Created {len(viz_paths)} visualizations")
    
    # Generate final output
    print("\n[9/9] Generating final JSON output...")
    output_gen = OutputGenerator()
    output = output_gen.generate_output(
        themes, temporal_data, anomalies, patterns, insights, viz_paths
    )
    
    # Save output
    output_path = 'output/results.json'
    Path('output').mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"✓ Saved results to {output_path}")
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)
    print(f"\nResults saved to: {output_path}")
    print(f"Visualizations saved to: output/")
    

if __name__ == "__main__":
    main()
