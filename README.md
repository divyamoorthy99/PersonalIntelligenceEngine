# Multi-Modal Life Pattern Personal Intelligence Pipeline

## Overview

This project implements an **end-to-end AI pipeline** that analyzes **30 days of multi-modal personal data** — including text journals, voice transcripts, and image captions — to uncover behavioral patterns, detect anomalies, track emotional cycles, and generate actionable insights.

The system emphasizes **interpretability, modularity, and responsible AI practices**, making it suitable for research demonstrations, technical interviews, and real-world experimentation with behavioral analytics.

---

## Key Features

* **Multi-Modal Intelligence**
  Unified semantic understanding across text, voice, and image data using transformer-based embeddings.

* **Life Theme Discovery**
  Automatically identifies 3–6 core behavioral themes using unsupervised clustering.

* **Temporal Pattern Analysis**
  Tracks how themes, mood, and behaviors evolve week-by-week.

* **Anomaly Detection**
  Flags unusual events such as stress surges, emotional dips, or fatigue spikes.

* **Cyclic Pattern Recognition**
  Detects recurring patterns (weekly cycles, day-of-week trends).

* **Insight Generation**
  Produces:
  * **Micro insights** (weekly)
  * **Macro insights** (30-day summary)
  * **Predictive insights** (near-future outlook)

* **Rich Visualizations**
  * 30-day multi-layer timeline (themes, mood, anomalies)
  * Weekly emotional radar comparison
  * Theme relationship network graph

---

## Project Structure
```
PersonalIntelligenceEngine/
├── data/
│   └── input_data.json          # 30-day synthetic dataset
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Data loading & preprocessing
│   ├── embeddings.py            # Multi-modal embedding generation
│   ├── clustering.py            # Theme clustering
│   ├── temporal_analysis.py     # Temporal trend analysis
│   ├── anomaly_detector.py      # Anomaly detection
│   ├── pattern_detector.py      # Cyclic pattern detection
│   ├── insight_generator.py     # Insight generation
│   ├── visualizer.py            # Visualization logic
│   └── output_generator.py      # Structured output creation
├── output/
│   ├── results.json             # Final structured insights
│   ├── timeline.png             # 30-day timeline visualization
│   ├── radar_chart.png          # Weekly radar chart
│   └── motif_graph.png          # Theme relationship network
├── main.py                      # Pipeline entry point
├── requirements.txt             # Python dependencies
└── README.md
```

---

## Setup Instructions

### Prerequisites
* Python **3.10+**
* pip (Python package manager)

### Installation

1. **Navigate to the project directory**

   ```bash
   cd PersonalIntelligenceEngine
   ```

2. **Create and activate a virtual environment (recommended)**

   ```bash
   python3 -m venv venv
  ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   Installed libraries include:
   * Data processing: `numpy`, `pandas`, `scipy`
   * ML algorithms: `scikit-learn`
   * Embeddings: `sentence-transformers`, `torch`, `transformers`
   * Visualization: `matplotlib`, `seaborn`, `networkx`

---

## Running the Pipeline

Execute the complete pipeline with:

```bash
python main.py
```

### Pipeline Execution Flow

1. Load the 30-day dataset from `data/input_data.json`
2. Generate unified multi-modal embeddings
3. Cluster entries into life themes
4. Perform weekly temporal analysis
5. Detect anomalies and cyclic patterns
6. Generate insights
7. Create visualizations
8. Save structured outputs

---

## Output Artifacts

After execution, the following outputs are generated:

### Structured Results

* **`output/results.json`**
  * Identified themes with keywords and confidence scores
  * Weekly summaries with micro-insights
  * Anomaly log with timestamps and descriptions
  * Pattern and cycle analysis
  * Macro and predictive insights
  * Safety and interpretation notes

### Visualizations
* **`timeline.png`** – 30-day multi-layer behavioral timeline
* **`radar_chart.png`** – Weekly emotional profile comparison
* **`motif_graph.png`** – Theme relationship and transition network

---

## Dataset Description
The input file `data/input_data.json` contains **30 daily entries**, each structured as:

```json
{
  "entry_id": "d_001",
  "date": "2025-10-01",
  "text": "Diary entry text...",
  "voice_transcript": "Voice note transcript...",
  "image_caption": "AI-generated image description...",
  "location_city": "City name"
}
```

### Dataset Characteristics
* **Themes**: Work, stress, health, social life, rest, personal growth, leisure
* **Temporal realism**: Weekday stress, weekend recovery patterns
* **Emotional variation**: Gradual trends, spikes, and dips
* **Anomalies**: Stress surges, confidence dips, fatigue events

> ⚠️ This dataset is **synthetic** and used purely for demonstration purposes.

---

## Design Rationale
### Multi-Modal Embeddings
* **Model**: `all-MiniLM-L6-v2` (Sentence-Transformers)
* Efficient, lightweight, and semantically rich
* Produces 384-dimensional embeddings

### Theme Clustering
* **Algorithm**: K-Means (default `k=5`)
* Interpretable and effective for latent theme discovery
* Enables representative sample extraction

### Temporal Analysis
* Weekly aggregation aligns with natural human cycles
* Supports trend classification: improving, declining, stable

### Anomaly Detection
* **Algorithm**: Isolation Forest
* Unsupervised and well-suited for high-dimensional embeddings
* Configurable sensitivity via contamination parameter

### Visualization Strategy
* **Timeline**: Temporal evolution at a glance
* **Radar Chart**: Multi-dimensional weekly comparison
* **Network Graph**: Theme relationships and transitions

---

## Customization Options

### Change Number of Themes
```python
clusterer = ThemeClusterer(n_clusters=6)
```

### Adjust Anomaly Sensitivity
```python
def __init__(self, contamination: float = 0.15)
```

### Switch Embedding Model
```python
def __init__(self, model_name: str = "all-mpnet-base-v2")
```

---

## Responsible AI Principles
This project follows responsible AI best practices:

1. **Non-diagnostic** – No medical claims or diagnoses
2. **Transparent** – Methods and limitations clearly documented
3. **Privacy-aware** – No identifying or sensitive personal data
4. **Interpretable** – Human-readable outputs and explanations
5. **Safety-first** – Risk detection with professional guidance notes
6. **Bias-conscious** – Neutral keyword design and interpretation

---

## Performance
* **Runtime**: ~30–60 seconds (including model load)
* **Memory Usage**: < 2 GB RAM
* **Scalability**: Easily extensible to 100+ days of data

---
contributor: Divya Moorthy @divyapeachi99@gmail.com

