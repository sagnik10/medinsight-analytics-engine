import os
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.neighbors import NearestNeighbors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
import warnings
warnings.filterwarnings("ignore")

start=time.time()

BASE_DIR=os.getcwd()
OUTPUT_DIR=os.path.join(BASE_DIR,"Output")
CHART_DIR=os.path.join(OUTPUT_DIR,"charts")
MODEL_DIR=os.path.join(OUTPUT_DIR,"models")
REC_DIR=os.path.join(OUTPUT_DIR,"recommendations")

os.makedirs(CHART_DIR,exist_ok=True)
os.makedirs(MODEL_DIR,exist_ok=True)
os.makedirs(REC_DIR,exist_ok=True)

INPUT_FILE="indian_diseases_dataset.csv"

df=pd.read_csv(INPUT_FILE)
df.columns=[c.lower().replace(" ","_") for c in df.columns]

initial_rows=len(df)
df=df.drop_duplicates()
duplicates_removed=initial_rows-len(df)

null_counts=df.isnull().sum()
cols_drop=[c for c in df.columns if null_counts[c]>4]
df=df.drop(columns=cols_drop)
df=df.dropna()

datetime_cols=[]
for c in df.columns:
    if "date" in c.lower() or "time" in c.lower():
        try:
            df[c]=pd.to_datetime(df[c])
            datetime_cols.append(c)
        except:
            pass

if len(datetime_cols)>0:
    df=df.sort_values(by=datetime_cols[0])

for c in df.columns:
    if df[c].dtype=="object":
        try:
            df[c]=pd.to_numeric(df[c])
        except:
            pass

numeric=df.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric)==0:
    raise ValueError("No numeric columns available after cleaning.")

df[numeric]=df[numeric].fillna(df[numeric].median())

DARK="#0b1220"
PANEL="#111827"
TEXT="#e5e7eb"
ACCENT="#22d3ee"
ACCENT2="#a78bfa"
ACCENT3="#34d399"

plt.rcParams.update({
"figure.facecolor":DARK,
"axes.facecolor":PANEL,
"text.color":TEXT,
"axes.labelcolor":TEXT,
"xtick.color":TEXT,
"ytick.color":TEXT,
"font.size":12
})

scaler=StandardScaler()
scaled=scaler.fit_transform(df[numeric])
pickle.dump(scaler,open(os.path.join(MODEL_DIR,"scaler.pkl"),"wb"))

pca=PCA(n_components=min(8,len(numeric)))
pca_data=pca.fit_transform(scaled)
pickle.dump(pca,open(os.path.join(MODEL_DIR,"pca.pkl"),"wb"))
explained_var=round(np.sum(pca.explained_variance_ratio_)*100,2)

charts_info=[]

fig,ax=plt.subplots(figsize=(14,8))
ax.plot(range(1,len(pca.explained_variance_ratio_)+1),
        np.cumsum(pca.explained_variance_ratio_),
        marker="o",linewidth=3,color=ACCENT)
ax.set_title("Cumulative PCA Explained Variance")
ax.set_xlabel("Principal Component Index")
ax.set_ylabel("Cumulative Explained Variance Ratio")
path=os.path.join(CHART_DIR,"pca_variance.png")
fig.savefig(path,dpi=300,bbox_inches="tight")
plt.close()
charts_info.append((path,
"Cumulative PCA Explained Variance",
"X-axis: Principal Component Index | Y-axis: Cumulative Explained Variance Ratio",
"Variance accumulation across principal components showing dimensional compression efficiency."))

if len(numeric)>=2:
    fig,ax=plt.subplots(figsize=(14,8))
    ax.scatter(pca_data[:,0],pca_data[:,1],s=20,color=ACCENT3,alpha=0.6)
    ax.set_title("PCA 2D Projection")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    path=os.path.join(CHART_DIR,"pca_projection.png")
    fig.savefig(path,dpi=300,bbox_inches="tight")
    plt.close()
    charts_info.append((path,
    "PCA 2D Projection",
    "X-axis: Principal Component 1 | Y-axis: Principal Component 2",
    "Reduced dimensional representation highlighting structural variation and clustering tendency."))

corr=df[numeric].corr()
fig,ax=plt.subplots(figsize=(14,10))
sns.heatmap(corr,cmap="viridis",ax=ax)
ax.set_title("Feature Correlation Matrix")
ax.set_xlabel("Features")
ax.set_ylabel("Features")
path=os.path.join(CHART_DIR,"correlation_matrix.png")
fig.savefig(path,dpi=300,bbox_inches="tight")
plt.close()
charts_info.append((path,
"Correlation Matrix",
"X-axis: Features | Y-axis: Features",
"Pairwise linear relationships between numeric variables to identify dependency structures."))

for col in numeric:
    bins=int(np.sqrt(len(df)))
    fig,ax=plt.subplots(figsize=(14,6))
    sns.histplot(df[col],bins=bins,kde=True,color=ACCENT,ax=ax)
    ax.set_title(f"Distribution of {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    path=os.path.join(CHART_DIR,f"distribution_{col}.png")
    fig.savefig(path,dpi=300,bbox_inches="tight")
    plt.close()
    charts_info.append((path,
    f"Distribution of {col}",
    f"X-axis: {col} | Y-axis: Frequency",
    "Histogram with adaptive binning revealing spread, skewness, and density variation."))

iso=IsolationForest(contamination=0.05,random_state=42)
anomaly_flags=iso.fit_predict(scaled)
df["anomaly_flag"]=anomaly_flags
anomaly_count=int((anomaly_flags==-1).sum())

if len(numeric)>=2:
    fig,ax=plt.subplots(figsize=(14,8))
    ax.scatter(pca_data[:,0],pca_data[:,1],c=anomaly_flags,cmap="coolwarm",s=20,alpha=0.7)
    ax.set_title("Isolation Forest Anomaly Detection")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    path=os.path.join(CHART_DIR,"anomaly_detection.png")
    fig.savefig(path,dpi=300,bbox_inches="tight")
    plt.close()
    charts_info.append((path,
    "Isolation Forest Anomaly Detection",
    "X-axis: Principal Component 1 | Y-axis: Principal Component 2",
    "Isolation Forest separation of anomalous observations in reduced space."))

kmeans=KMeans(n_clusters=4,n_init=20,random_state=42)
clusters=kmeans.fit_predict(scaled)
df["cluster"]=clusters
sil=round(silhouette_score(scaled,clusters),3)

if len(numeric)>=2:
    fig,ax=plt.subplots(figsize=(14,8))
    ax.scatter(pca_data[:,0],pca_data[:,1],c=clusters,cmap="viridis",s=20,alpha=0.7)
    ax.set_title("KMeans Clustering Projection")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    path=os.path.join(CHART_DIR,"cluster_projection.png")
    fig.savefig(path,dpi=300,bbox_inches="tight")
    plt.close()
    charts_info.append((path,
    "KMeans Clustering Projection",
    "X-axis: Principal Component 1 | Y-axis: Principal Component 2",
    "Segmentation of data into four structural clusters."))

fig,ax=plt.subplots(figsize=(10,6))
df["cluster"].value_counts().sort_index().plot(kind="bar",ax=ax,color=ACCENT3)
ax.set_title("Cluster Distribution")
ax.set_xlabel("Cluster Label")
ax.set_ylabel("Number of Records")
path=os.path.join(CHART_DIR,"cluster_distribution.png")
fig.savefig(path,dpi=300,bbox_inches="tight")
plt.close()
charts_info.append((path,
"Cluster Distribution",
"X-axis: Cluster Label | Y-axis: Number of Records",
"Population distribution across discovered clusters."))

if len(numeric)>1:
    target=numeric[0]
    mi=mutual_info_regression(df[numeric],df[target])
    imp=pd.Series(mi,index=numeric).sort_values()
    fig,ax=plt.subplots(figsize=(14,8))
    imp.plot(kind="barh",ax=ax,color=ACCENT2)
    ax.set_title(f"Feature Importance Relative to {target}")
    ax.set_xlabel("Mutual Information Score")
    path=os.path.join(CHART_DIR,"feature_importance.png")
    fig.savefig(path,dpi=300,bbox_inches="tight")
    plt.close()
    charts_info.append((path,
    f"Feature Importance Relative to {target}",
    "X-axis: Mutual Information Score | Y-axis: Features",
    "Nonlinear dependency strength between variables and reference feature."))

nn=NearestNeighbors(n_neighbors=6,metric="cosine",algorithm="brute")
nn.fit(scaled)
distances,indices=nn.kneighbors(scaled)

recommendations=[]
for i in range(len(df)):
    for j in range(1,6):
        recommendations.append({
            "source_index":i,
            "recommended_index":indices[i][j],
            "similarity":round(1-distances[i][j],4)
        })

rec_df=pd.DataFrame(recommendations)
rec_df.to_csv(os.path.join(REC_DIR,"similarity_recommendations.csv"),index=False)

execution=round(time.time()-start,2)

styles=getSampleStyleSheet()
title_style=ParagraphStyle(name="title",fontSize=28,leading=34,alignment=1,
                           textColor=HexColor("#22d3ee"),spaceAfter=30)
body_style=ParagraphStyle(name="body",fontSize=11,leading=18,spaceAfter=15)
heading_style=ParagraphStyle(name="heading",fontSize=18,leading=24,
                             textColor=HexColor("#a78bfa"),spaceAfter=15)

doc=SimpleDocTemplate(os.path.join(OUTPUT_DIR,"Analytics_Report.pdf"),
                      leftMargin=50,rightMargin=50,topMargin=50,bottomMargin=50)

elements=[]
elements.append(Paragraph("Indian Diseases Dataset Analytics Report",title_style))

summary_text=f"""
Dataset: indian_diseases_dataset.csv<br/>
Total Records After Cleaning: {len(df)}<br/>
Exact Duplicates Removed: {duplicates_removed}<br/>
Columns Dropped (>4 Nulls): {len(cols_drop)}<br/>
Numeric Features Used: {len(numeric)}<br/>
Detected Anomalies: {anomaly_count}<br/>
Silhouette Score: {sil}<br/>
Total PCA Explained Variance: {explained_var}%<br/>
Execution Time: {execution} seconds
"""
elements.append(Paragraph(summary_text,body_style))
elements.append(PageBreak())

for path,title,axes_text,explanation in charts_info:
    elements.append(Paragraph(title,heading_style))
    elements.append(Spacer(1,10))
    elements.append(Image(path,width=6.5*inch,height=4.2*inch))
    elements.append(Spacer(1,10))
    elements.append(Paragraph(f"<b>Axes:</b> {axes_text}",body_style))
    elements.append(Paragraph(f"<b>Explanation:</b> {explanation}",body_style))
    elements.append(PageBreak())

doc.build(elements)

print("Complete")
print("Execution Time:",execution)