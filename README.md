# MedInsight Analytics Engine

An end-to-end healthcare analytics pipeline for structured patient disease datasets. This repository performs automated data cleaning, anomaly detection, clustering, dimensionality reduction, similarity modeling, and generates a structured PDF analytics report with visualizations.

---

## Project Structure

```
INDIA_PATIENT_DISEASE_RECORDS/
│
├── Analyzer.py
├── indian_diseases_dataset_analysis.csv
│
├── Output/
│   ├── Analytics_Report.pdf
│   │
│   ├── charts/
│   │   ├── anomaly_detection.png
│   │   ├── cluster_distribution.png
│   │   ├── cluster_projection.png
│   │   ├── correlation_matrix.png
│   │   ├── distribution_age.png
│   │   ├── distribution_bmi.png
│   │   ├── distribution_days_hospitalized.png
│   │   ├── distribution_death_flag.png
│   │   ├── distribution_treatment_cost_inr.png
│   │   ├── distribution_year.png
│   │   ├── feature_importance.png
│   │   ├── pca_projection.png
│   │   └── pca_variance.png
│   │
│   ├── models/
│   │   ├── scaler.pkl
│   │   └── pca.pkl
│   │
│   └── recommendations/
│       └── similarity_recommendations.csv
```

---

## Features

### Data Cleaning Pipeline
- Removes exact duplicate records
- Drops columns with excessive null values
- Removes remaining null rows
- Converts datetime columns automatically
- Sorts dataset chronologically
- Dynamically detects numeric features

### Exploratory Data Analysis
- Correlation heatmap
- Feature distribution plots
- PCA explained variance
- PCA 2D projection

### Dimensionality Reduction
- Standard scaling
- Principal Component Analysis
- Explained variance reporting

### Anomaly Detection
- Isolation Forest
- PCA-based anomaly visualization
- Anomaly count reporting

### Clustering
- KMeans clustering
- Cluster projection visualization
- Cluster distribution chart
- Silhouette score evaluation

### Feature Importance
- Mutual Information based ranking
- Non-linear dependency analysis

### Similarity Engine
- Nearest Neighbors with cosine similarity
- Top 5 similar record recommendations
- Exported similarity CSV

### Automated Report Generation
- Structured PDF analytics report
- Embedded charts
- Summary metrics
- Chart explanations

---

## Generated Outputs

Visualizations are stored in:
`Output/charts/`

Model artifacts are stored in:
`Output/models/`

Similarity results:
`Output/recommendations/similarity_recommendations.csv`

Final PDF report:
`Output/Analytics_Report.pdf`

---

## Installation

```
pip install numpy pandas matplotlib seaborn scikit-learn reportlab
```

---

## Usage

Run locally:
```
python Analyzer.py
```

Run on Kaggle:
Upload the script, attach the dataset, execute the notebook. Outputs will be saved to `/kaggle/working/Output`.

---

## Pipeline Flow

```
Raw Dataset
    ↓
Cleaning and Null Handling
    ↓
Datetime Sorting
    ↓
Feature Scaling
    ↓
PCA
    ↓
Anomaly Detection
    ↓
Clustering
    ↓
Feature Importance
    ↓
Similarity Modeling
    ↓
PDF Report Generation
```

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- ReportLab

---

## License

This project is licensed under the Apache License 2.0.

See the `LICENSE` file for full details.
