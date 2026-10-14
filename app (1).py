from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ============================================================
# PAGE CONFIGURATION AND DARK THEME
# ============================================================
st.set_page_config(page_title="Customer Intelligence | Segmentation", page_icon="◈", layout="wide")
st.markdown("""
<style>
.stApp {background: #10131b; color: #edf0f7;}
[data-testid="stHeader"] {background: rgba(16,19,27,0.92);}
[data-testid="stSidebar"] {background: #171b26;}
.block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1500px;}
.hero {padding: 1.5rem 1.7rem; border: 1px solid #30384a; border-radius: 18px;
       background: linear-gradient(120deg,#20283a,#171b27 70%); margin-bottom: 1.1rem;}
.hero h1 {margin: 0 0 .35rem 0; color: #f7f8fc; font-size: 2.1rem;}
.hero p {margin: 0; color: #b7c0d4;}
.panel {padding: 1rem 1.1rem; background: #171c28; border: 1px solid #2c3446; border-radius: 14px;}
.small-note {color:#9ca8bf; font-size:.82rem;}
section[data-testid="stSidebar"] label, .stMarkdown, p, label {color: #e4e8f1;}
[data-testid="stMetric"] {background:#171c28; padding:14px 16px; border:1px solid #2c3446; border-radius:13px;}
[data-testid="stMetricLabel"] {color:#aeb9cf;}
[data-testid="stMetricValue"] {color:#f7f8fc;}
</style>
""", unsafe_allow_html=True)


def section(title, subtitle=""):
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)


def read_upload(uploaded_file, sheet_name=None):
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".csv":
        try:
            return pd.read_csv(uploaded_file), "CSV"
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding="latin-1"), "CSV"
    if suffix in (".xlsx", ".xls", ".xlsm"):
        xls = pd.ExcelFile(uploaded_file)
        chosen = sheet_name if sheet_name in xls.sheet_names else xls.sheet_names[0]
        return pd.read_excel(xls, sheet_name=chosen), chosen
    raise ValueError("Please upload a CSV or Excel file.")


def build_conclusion(result, profiles, noise_count, numeric_cols, categorical_cols, silhouette, algorithm):
    lines = []
    n = len(result)
    cluster_labels = [x for x in result["Cluster"].unique() if x != -1]
    lines.append(f"The analysis evaluated {n:,} rows using {len(numeric_cols)} numeric and {len(categorical_cols)} categorical feature(s).")
    lines.append(f"{len(cluster_labels)} cluster(s) were identified; {noise_count:,} row(s) were labelled as noise/unassigned.")
    if silhouette is not None:
        lines.append(f"The silhouette score on sampled non-noise records was {silhouette:.3f}; interpret it alongside the cluster plot and domain knowledge.")
    else:
        lines.append("A silhouette score was not available because there were too few non-noise clusters or records to calculate it reliably.")
    if not profiles.empty and numeric_cols:
        usable = [c for c in numeric_cols if c in profiles.columns and not any(token in str(c).lower() for token in ("id", "customer number", "customer_no", "index", "code", "zip", "postal"))]
        if usable:
            col = max(usable, key=lambda c: profiles[c].nunique(dropna=True))
            hi = profiles[col].idxmax()
            lo = profiles[col].idxmin()
            if hi != lo:
                lines.append(f"Cluster {hi} has the highest average {col}; Cluster {lo} has the lowest average {col} among the detected clusters.")
    lines.append("Cluster labels describe similarity patterns, not proven customer types or causal relationships. Validate the findings before business action.")
    return lines


def infer_segment_names(profiles, numeric_cols, cluster_counts):
    """Infer cautious, human-readable segment labels from cluster averages."""
    if profiles.empty:
        return {}, {}
    def find_col(words):
        for col in profiles.columns:
            name = str(col).lower().replace("_", " ").replace("-", " ")
            if any(word in name for word in words):
                return col
        return None
    spend = find_col(["spend", "spending", "monetary", "revenue", "sales", "amount"])
    purchases = find_col(["purchase", "purchases", "orders", "transactions", "frequency"])
    campaign = find_col(["campaign", "accepted", "response", "conversion"])
    income = find_col(["income", "salary", "earnings"])
    usable = [c for c in numeric_cols if c in profiles.columns and
              not any(t in str(c).lower() for t in ("id", "customer number", "customer_no", "index", "code", "zip", "postal"))]
    def level(col, cluster_id):
        if not col or col not in profiles.columns:
            return "middle"
        vals = pd.to_numeric(profiles[col], errors="coerce").dropna()
        if len(vals) < 2 or cluster_id not in vals.index:
            return "middle"
        rank = vals.rank(pct=True).loc[cluster_id]
        return "high" if rank >= .75 else ("low" if rank <= .25 else "middle")
    fallback = None
    spreads = {}
    for col in usable:
        vals = pd.to_numeric(profiles[col], errors="coerce").dropna()
        if len(vals) > 1:
            spreads[col] = float(vals.max() - vals.min()) / (abs(float(vals.mean())) + 1e-9)
    if spreads:
        fallback = max(spreads, key=spreads.get)
    names, explanations = {}, {}
    for cid in profiles.index:
        s = level(spend, cid) if spend else None
        p = level(purchases, cid) if purchases else None
        c = level(campaign, cid) if campaign else None
        i = level(income, cid) if income else None
        if s == "high" and p == "high":
            name, traits = "High-Value Frequent Buyers", f"higher average {spend} and {purchases}"
        elif s == "high":
            name, traits = "High-Spending Customers", f"higher average {spend}"
        elif s == "low" and p == "low":
            name, traits = "Low-Spending Occasional Buyers", f"lower average {spend} and {purchases}"
        elif s == "low":
            name, traits = "Lower-Spending Customers", f"lower average {spend}"
        elif p == "high":
            name, traits = "Frequent Buyers", f"higher average {purchases}"
        elif c == "high":
            name, traits = "Campaign-Responsive Customers", f"higher average {campaign}"
        elif i == "high":
            name, traits = "Higher-Income Customers", f"higher average {income}"
        elif i == "low":
            name, traits = "Lower-Income Customers", f"lower average {income}"
        elif fallback:
            lev = level(fallback, cid)
            name = (f"High {fallback} Segment" if lev == "high" else
                    f"Low {fallback} Segment" if lev == "low" else f"Typical {fallback} Segment")
            traits = f"{lev}-range average {fallback} compared with other clusters"
        else:
            name, traits = "Distinct Behaviour Group", "a different combination of selected features"
        if campaign and c == "high" and "Campaign-Responsive" not in name:
            traits += f"; also higher average {campaign}"
        count = int(cluster_counts.get(cid, 0))
        names[int(cid)] = name
        explanations[int(cid)] = f"{count:,} records. This group shows {traits}."
    return names, explanations


st.markdown("""
<div class="hero">
  <h1>◈ Customer Intelligence Dashboard</h1>
  <p>Adaptive spreadsheet analysis · Customer segmentation · Behaviour patterns · Business recommendations</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## Analysis controls")
    st.write("Upload a CSV or Excel workbook to begin.")
    uploaded_file = st.file_uploader("Customer dataset", type=["csv", "xlsx", "xls", "xlsm"])
    sheet_name = None
    if uploaded_file and Path(uploaded_file.name).suffix.lower() in (".xlsx", ".xls", ".xlsm"):
        try:
            uploaded_file.seek(0)
            sheet_names = pd.ExcelFile(uploaded_file).sheet_names
            sheet_name = st.selectbox("Worksheet", sheet_names)
        except Exception:
            sheet_name = None
    algorithm = st.selectbox("Clustering algorithm", ["DBSCAN", "K-Means"], key="clustering_algorithm")
    if algorithm == "DBSCAN":
        eps = st.slider(
            "DBSCAN epsilon",
            min_value=0.5,
            max_value=10.0,
            value=2.0,
            step=0.5,
            help="Distance threshold in the prepared feature space. Larger values can connect more points.",
            key="dbscan_eps",
        )
        min_samples = st.slider(
            "DBSCAN minimum samples",
            min_value=2,
            max_value=30,
            value=5,
            step=1,
            help="Minimum neighbourhood size for a core point, including the point itself.",
            key="dbscan_min_samples",
        )
    else:
        k_clusters = st.slider(
            "K-Means number of clusters",
            min_value=2,
            max_value=15,
            value=2,
            step=1,
            key="kmeans_n_clusters",
        )
        kmeans_n_init = st.slider(
            "K-Means n_init",
            min_value=1,
            max_value=30,
            value=10,
            step=1,
            help="Number of different centroid initialisations. K-Means keeps the best run by inertia.",
            key="kmeans_n_init",
        )
    max_rows = st.selectbox("Maximum rows to analyse", [1000, 5000, 10000, 25000, 50000], index=3)

if not uploaded_file:
    st.info("Upload your customer dataset in the sidebar to generate the analysis.")
    st.markdown("""
    <div class="panel">
    <b>What this dashboard provides</b><br><br>
    • Automatic column-type detection and preprocessing<br>
    • Numeric and categorical feature selection<br>
    • DBSCAN density clustering or K-Means comparison<br>
    • PCA behaviour map, cluster profiles, and distribution charts<br>
    • Data quality diagnostics and downloadable results<br>
    • Dataset-based conclusions and cautious business recommendations
    </div>
    """, unsafe_allow_html=True)
    st.stop()

try:
    uploaded_file.seek(0)
    raw_data, sheet_used = read_upload(uploaded_file, sheet_name)
except Exception as exc:
    st.error(f"Could not read this file: {exc}")
    st.stop()

if raw_data.empty:
    st.error("The uploaded file has no data rows.")
    st.stop()

# Clean headers and remove fully empty rows/columns without assuming fixed positions.
raw_data.columns = [str(c).strip() if str(c).strip() else f"Unnamed_{i+1}" for i, c in enumerate(raw_data.columns)]
raw_data = raw_data.loc[:, ~raw_data.columns.duplicated()].copy()
raw_data = raw_data.dropna(axis=0, how="all").dropna(axis=1, how="all")
if raw_data.empty or raw_data.shape[1] == 0:
    st.error("No usable data remains after removing completely empty rows and columns.")
    st.stop()

if len(raw_data) > max_rows:
    st.warning(f"The file contains {len(raw_data):,} rows. Analysing the first {max_rows:,} rows for performance.")
    raw_data = raw_data.head(max_rows).copy()

# Treat numeric columns with sufficiently numeric values as numeric; other columns are categorical.
working = raw_data.copy()

# Add business-friendly features when the uploaded file matches the marketing
# campaign dataset. These are derived by column names, not column positions.
if "Year_Birth" in working.columns and "Age" not in working.columns:
    birth_year = pd.to_numeric(working["Year_Birth"], errors="coerce")
    working["Age"] = 2026 - birth_year
if "Kidhome" in working.columns and "Teenhome" in working.columns and "Total_Children" not in working.columns:
    working["Total_Children"] = (
        pd.to_numeric(working["Kidhome"], errors="coerce").fillna(0)
        + pd.to_numeric(working["Teenhome"], errors="coerce").fillna(0)
    )
spending_cols = [c for c in working.columns if str(c).lower().startswith("mnt") and pd.api.types.is_numeric_dtype(working[c])]
if spending_cols and "Total_Spending" not in working.columns:
    working["Total_Spending"] = working[spending_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
purchase_cols = [c for c in ["NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"] if c in working.columns]
if purchase_cols and "Total_Purchases" not in working.columns:
    working["Total_Purchases"] = working[purchase_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
campaign_cols = [c for c in working.columns if str(c).lower().startswith("acceptedcmp")]
if "Response" in working.columns:
    campaign_cols = campaign_cols + ["Response"]
if campaign_cols and "Total_Campaign_Acceptance" not in working.columns:
    working["Total_Campaign_Acceptance"] = working[campaign_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)

for col in working.columns:
    if working[col].dtype == "object":
        converted = pd.to_numeric(working[col].astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce")
        non_missing = working[col].notna().sum()
        if non_missing and converted.notna().sum() / non_missing >= 0.9:
            working[col] = converted

numeric_candidates = [c for c in working.columns if pd.api.types.is_numeric_dtype(working[c]) and working[c].nunique(dropna=True) > 1]
categorical_candidates = [c for c in working.columns if c not in numeric_candidates and working[c].nunique(dropna=True) > 1]

section("01 · Dataset overview", "Inspect the uploaded data before selecting the features used for clustering.")
a, b, c, d = st.columns(4)
a.metric("Rows analysed", f"{len(working):,}")
b.metric("Columns available", f"{working.shape[1]:,}")
c.metric("Missing cells", f"{int(working.isna().sum().sum()):,}")
d.metric("Duplicate rows", f"{int(working.duplicated().sum()):,}")
with st.expander("Preview uploaded data", expanded=False):
    st.dataframe(raw_data.head(20), use_container_width=True)
    dtype_table = pd.DataFrame({"Column": working.columns, "Data type": [str(working[x].dtype) for x in working.columns],
                                "Missing values": [int(working[x].isna().sum()) for x in working.columns],
                                "Unique values": [int(working[x].nunique(dropna=True)) for x in working.columns]})
    st.dataframe(dtype_table, use_container_width=True, hide_index=True)

section("02 · Feature selection", "Choose columns that describe customer behaviour. ID-like columns are excluded from the default selection.")
# Avoid identifiers and constant configuration fields by default. For the
# marketing campaign dataset, prioritize interpretable customer behaviour.
identifier_tokens = ("id", "customer number", "customer_no", "index", "postal", "zip code")
def_num = [
    c for c in numeric_candidates
    if not any(token in str(c).lower().replace("_", " ") for token in identifier_tokens)
    and str(c).lower() not in ("z_costcontact", "z_revenue")
]
preferred_marketing_features = [
    "Age", "Income", "Total_Children", "Recency", "Total_Spending",
    "Total_Purchases", "Total_Campaign_Acceptance", "NumDealsPurchases",
    "NumWebVisitsMonth"
]
if "Total_Spending" in working.columns and "Total_Purchases" in working.columns:
    def_num = [c for c in preferred_marketing_features if c in numeric_candidates]
    def_cat = [c for c in ["Education", "Marital_Status"] if c in categorical_candidates]
else:
    def_cat = categorical_candidates[:]
with st.expander("Choose analysis features", expanded=True):
    selected_numeric = st.multiselect("Numeric features", numeric_candidates, default=def_num,
                                      help="Numeric values are median-imputed and standardized.")
    selected_categorical = st.multiselect("Categorical features", categorical_candidates, default=def_cat,
                                          help="Categories are most-frequent-imputed and one-hot encoded.")
selected_features = selected_numeric + selected_categorical
selected_id_like = [
    c for c in selected_features
    if any(token in str(c).lower().replace("_", " ") for token in identifier_tokens)
]
if selected_id_like:
    st.warning(
        "These selected columns look like identifiers rather than customer behaviour: "
        + ", ".join(selected_id_like)
        + ". Consider removing them so cluster profiles and distances are not driven by ID values."
    )
if not selected_features:
    st.warning("Select at least one numeric or categorical feature to continue.")
    st.stop()

# Exclude constant columns and guard against excessive one-hot expansion.
selected_numeric = [c for c in selected_numeric if working[c].nunique(dropna=True) > 1]
selected_categorical = [c for c in selected_categorical if working[c].nunique(dropna=True) > 1]
selected_features = selected_numeric + selected_categorical
if not selected_features:
    st.error("The selected columns do not contain enough variation for clustering.")
    st.stop()
if len(selected_categorical) > 20:
    st.warning("Many categorical columns were selected. One-hot encoding may create a high-dimensional feature space.")

X_raw = working[selected_features].copy()
numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
try:
    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=100)
except TypeError:
    try:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse=False, max_categories=100)
    except TypeError:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)
cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", onehot)])
transformers = []
if selected_numeric:
    transformers.append(("numeric", numeric_pipe, selected_numeric))
if selected_categorical:
    transformers.append(("categorical", cat_pipe, selected_categorical))
preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

try:
    X = preprocessor.fit_transform(X_raw)
    X = np.asarray(X, dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
except TypeError:
    # Compatibility fallback for older scikit-learn versions.
    if selected_categorical:
        cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                             ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))])
        transformers = []
        if selected_numeric:
            transformers.append(("numeric", numeric_pipe, selected_numeric))
        transformers.append(("categorical", cat_pipe, selected_categorical))
        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
        X = np.asarray(preprocessor.fit_transform(X_raw), dtype=float)
    else:
        X = np.asarray(preprocessor.fit_transform(X_raw), dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
except Exception as exc:
    st.error(f"Feature preprocessing failed: {exc}")
    st.stop()

if X.shape[0] < 3 or X.shape[1] < 1:
    st.error("At least 3 usable rows and one transformed feature are needed for clustering.")
    st.stop()

# Fit selected clustering model.
try:
    if algorithm == "DBSCAN":
        cluster_model = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
        labels = cluster_model.fit_predict(X)
        model_name = "DBSCAN"
    else:
        n_k = min(k_clusters, len(working) - 1)
        cluster_model = KMeans(n_clusters=max(2, n_k), random_state=42, n_init=kmeans_n_init)
        labels = cluster_model.fit_predict(X)
        model_name = "K-Means"
except Exception as exc:
    st.error(f"Clustering failed: {exc}")
    st.stop()

result = raw_data.copy().reset_index(drop=True)
result["Cluster"] = labels
noise_count = int(np.sum(labels == -1)) if model_name == "DBSCAN" else 0
noise_pct = noise_count / len(labels) * 100 if len(labels) else 0.0
cluster_ids = sorted(int(x) for x in np.unique(labels) if x != -1)
n_clusters = len(cluster_ids)
cluster_names = {x: f"Cluster {x}" for x in cluster_ids}

silhouette = None
mask = labels != -1 if model_name == "DBSCAN" else np.ones(len(labels), dtype=bool)
valid_labels = labels[mask]
if len(np.unique(valid_labels)) >= 2 and len(valid_labels) > len(np.unique(valid_labels)) and len(valid_labels) <= 10000:
    try:
        silhouette = float(silhouette_score(X[mask], valid_labels, sample_size=min(3000, len(valid_labels)), random_state=42))
    except Exception:
        silhouette = None

# Create compact profile table from numeric selected fields and category modes.
profile_data = working.copy()
profile_data["Cluster"] = labels
non_noise = profile_data[profile_data["Cluster"] != -1] if model_name == "DBSCAN" else profile_data
profiles = pd.DataFrame()
if not non_noise.empty:
    profile_parts = []
    if selected_numeric:
        profile_parts.append(non_noise.groupby("Cluster")[selected_numeric].mean(numeric_only=True))
    if selected_categorical:
        modes = non_noise.groupby("Cluster")[selected_categorical].agg(
            lambda s: s.mode(dropna=True).iloc[0] if not s.mode(dropna=True).empty else "Unknown")
        profile_parts.append(modes)
    if profile_parts:
        profiles = pd.concat(profile_parts, axis=1)


cluster_counts = pd.Series(labels).value_counts().to_dict()
cluster_names, cluster_explanations = infer_segment_names(profiles, selected_numeric, cluster_counts)
cluster_names[-1] = "Noise / Unassigned"
cluster_explanations[-1] = "DBSCAN did not assign these records to a dense cluster; this does not automatically mean they are bad customers."
result["Customer Segment"] = [cluster_names.get(int(x), f"Cluster {int(x)}") for x in labels]

section("03 · Segmentation results", "Cluster labels are calculated from the selected features after preprocessing.")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Algorithm", model_name)
m2.metric("Clusters found", n_clusters)
m3.metric("Noise / unassigned", f"{noise_count:,}", f"{noise_pct:.1f}% of rows" if model_name == "DBSCAN" else None)
m4.metric("Silhouette score", f"{silhouette:.3f}" if silhouette is not None else "N/A")

if model_name == "DBSCAN" and n_clusters < 2:
    st.warning(
        "DBSCAN found fewer than two customer clusters, so it cannot provide a useful segment comparison yet. "
        "Try a smaller epsilon in small steps (for example 1.0, 1.5, 2.0) or adjust minimum samples. "
        "If DBSCAN continues to find one cluster, switch to K-Means and test 3–6 clusters, then compare the customer profiles."
    )

left, right = st.columns([1.2, 1])
with left:
    section("Cluster distribution")
    counts = pd.Series(labels).value_counts().sort_index()
    distribution = pd.DataFrame({"Cluster": [cluster_names.get(int(i), f"Cluster {i}") for i in counts.index], "Rows": counts.values})
    fig_dist = px.bar(distribution, x="Cluster", y="Rows", color="Cluster", text="Rows", title="Customers per cluster", template="plotly_dark")
    fig_dist.update_layout(margin=dict(l=10, r=10, t=55, b=10), height=360, showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_dist, use_container_width=True)
with right:
    section("Behaviour map (PCA)")
    try:
        if X.shape[1] >= 2:
            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(X)
        else:
            coords = np.column_stack([X[:, 0], np.zeros(len(X))])
        plot_df = pd.DataFrame({"Projection 1": coords[:, 0], "Projection 2": coords[:, 1],
                                "Cluster": [cluster_names.get(int(x), f"Cluster {x}") for x in labels]})
        hover_cols = []
        for col in selected_features[:4]:
            if col in working.columns:
                plot_df[col] = working[col].astype(str).reset_index(drop=True)
                hover_cols.append(col)
        fig_scatter = px.scatter(plot_df, x="Projection 1", y="Projection 2", color="Cluster", hover_data=hover_cols,
                                 title="2D PCA projection", template="plotly_dark", opacity=0.82)
        fig_scatter.update_layout(margin=dict(l=10, r=10, t=55, b=10), height=360,
                                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.caption("PCA is used here for visualization. Clustering was performed in the prepared feature space.")
    except Exception as exc:
        st.warning(f"Could not create the PCA plot: {exc}")

section("04 · Segment profiles", "Compare numeric averages and categorical modes for each non-noise cluster.")
if not profiles.empty:
    profile_display = profiles.copy()
    profile_display.index = [f"Cluster {int(i)} — {cluster_names.get(int(i), 'Distinct Behaviour Group')}" for i in profile_display.index]
    st.dataframe(profile_display.round(2), use_container_width=True)
    numeric_profile_cols = [
        c for c in profiles.columns
        if c in selected_numeric and pd.api.types.is_numeric_dtype(profiles[c])
        and not any(token in str(c).lower().replace("_", " ") for token in identifier_tokens)
    ]
    if numeric_profile_cols:
        metric_to_plot = st.selectbox("Choose a numeric feature to compare", numeric_profile_cols)
        bar_data = profiles[[metric_to_plot]].reset_index().rename(columns={"index": "Cluster"})
        bar_data["Cluster"] = bar_data["Cluster"].map(lambda x: f"Cluster {int(x)} — {cluster_names.get(int(x), 'Distinct Behaviour Group')}")
        fig_profile = px.bar(bar_data, x="Cluster", y=metric_to_plot, text_auto=".3s",
                             title=f"Average {metric_to_plot} by cluster", template="plotly_dark")
        fig_profile.update_layout(margin=dict(l=10, r=10, t=55, b=10), height=340,
                                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_profile, use_container_width=True)
else:
    st.info("No non-noise cluster profiles are available. Adjust DBSCAN settings or review selected features.")


section("04A · Customer type interpretation", "Human-readable labels are inferred from actual cluster averages.")
if cluster_names:
    for cid in cluster_ids:
        name = cluster_names.get(cid, f"Cluster {cid}")
        count = int(cluster_counts.get(cid, 0))
        share = count / len(labels) * 100
        st.markdown(
            f"<div class='panel' style='margin-bottom:10px'><b>{name}</b>"
            f"<br><span class='small-note'>Cluster {cid} · {count:,} records · {share:.1f}% of dataset</span>"
            f"<br><br>{cluster_explanations.get(cid, 'Review the cluster profile to understand this group.')}</div>",
            unsafe_allow_html=True
        )
    st.caption("These labels are inferred from column names and relative cluster averages. Verify them against the profile table before making business decisions.")
else:
    st.info("Cluster interpretations will appear when cluster profiles are available.")

section("05 · Data quality and diagnostics", "Review feature usage, missing data, and cluster distribution.")
q1, q2 = st.columns(2)
with q1:
    feature_summary = pd.DataFrame([
        {"Type": "Numeric", "Count": len(selected_numeric), "Columns": ", ".join(selected_numeric) or "None"},
        {"Type": "Categorical", "Count": len(selected_categorical), "Columns": ", ".join(selected_categorical) or "None"},
    ])
    st.markdown("**Feature usage**")
    st.dataframe(feature_summary, use_container_width=True, hide_index=True)
with q2:
    st.markdown("**Cluster distribution**")
    dist_rows = []
    for label in sorted(np.unique(labels)):
        count = int((labels == label).sum())
        dist_rows.append({"Cluster": cluster_names.get(int(label), f"Cluster {int(label)}"),
                          "Rows": count, "Share (%)": round(count / len(labels) * 100, 2)})
    st.dataframe(pd.DataFrame(dist_rows), use_container_width=True, hide_index=True)

section("06 · Model conclusion and business recommendations", "These observations are generated from the uploaded dataset and current model output.")
for line in build_conclusion(result, profiles, noise_count, selected_numeric, selected_categorical, silhouette, model_name):
    st.markdown(f"- {line}")
recommendations = []
if n_clusters == 0:
    recommendations.append("Adjust DBSCAN epsilon/minimum samples or compare results with K-Means before drawing segment-level conclusions.")
else:
    if model_name == "DBSCAN" and noise_pct > 30:
        recommendations.append("Review the high noise share. These records may represent sparse behaviour patterns, not necessarily bad data.")
    spend_col = next((c for c in profiles.columns if any(k in str(c).lower() for k in ("spend", "revenue", "amount", "sales", "purchase_value"))), None)
    if spend_col and not profiles.empty:
        high, low = profiles[spend_col].idxmax(), profiles[spend_col].idxmin()
        if high != low:
            recommendations.append(f"Explore retention or loyalty strategies for Cluster {high}, with the highest average {spend_col}.")
            recommendations.append(f"Explore product fit and targeted promotions for Cluster {low}, with the lowest average {spend_col}.")
    if not recommendations:
        recommendations.append("Review each segment profile and test a distinct message or service approach for each group.")
recommendations.append("Measure conversion, revenue, repeat purchases, and retention in controlled campaigns before claiming business impact.")
for item in recommendations:
    st.markdown(f"- {item}")
st.markdown("<div class='panel'><b>Executive takeaway</b><br><br>The model identifies patterns in the selected columns. Cluster labels are descriptive, not business truths by themselves. Validate interpretations with domain knowledge and measured outcomes before acting on them.</div>", unsafe_allow_html=True)

section("07 · Export results", "Download row-level cluster assignments and a compact analysis summary.")
download_left, download_right = st.columns(2)
with download_left:
    st.download_button("Download row-level results (CSV)", data=result.to_csv(index=False).encode("utf-8-sig"),
                       file_name="customer_segmentation_results.csv", mime="text/csv", use_container_width=True)
with download_right:
    summary_export = pd.DataFrame({"Metric": ["Source file", "Worksheet", "Rows analysed", "Original columns", "Features used",
                                               "Numeric features", "Categorical features", "Algorithm", "Clusters found",
                                               "Noise / unassigned rows", "Noise percentage", "Silhouette score"],
                                   "Value": [uploaded_file.name, str(sheet_used), len(raw_data), len(raw_data.columns), len(selected_features),
                                             len(selected_numeric), len(selected_categorical), model_name, n_clusters, noise_count,
                                             round(noise_pct, 2), round(silhouette, 4) if silhouette is not None else "Not available"]})
    st.download_button("Download analysis summary (CSV)", data=summary_export.to_csv(index=False).encode("utf-8-sig"),
                       file_name="segmentation_analysis_summary.csv", mime="text/csv", use_container_width=True)

st.markdown("---")
st.markdown("<p class='small-note'>Customer Intelligence · Adaptive spreadsheet analysis · Python · Pandas · Scikit-learn · Streamlit · Plotly</p>", unsafe_allow_html=True)
