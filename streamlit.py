import streamlit as st
from streamlit_lottie import st_lottie
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="IAnalyz - Customer Analytics",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Chargement d'une animation Lottie (pour l'accueil)
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# CSS personnalisé pour améliorer l'UX
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Chargement du CSS
local_css("style.css")

# Animation Lottie pour l'accueil
lottie_url = "https://assets9.lottiefiles.com/packages/lf20_0yfsb3a5.json"
lottie_animation = load_lottieurl(lottie_url)

# Fonction pour télécharger un fichier depuis Google Drive
def download_from_drive(drive_url):
    try:
        if '/d/' in drive_url:
            file_id = drive_url.split('/d/')[1].split('/')[0]
        elif 'id=' in drive_url:
            file_id = drive_url.split('id=')[1].split('&')[0]
        else:
            return None, "URL Google Drive invalide"

        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = requests.get(download_url)
        if response.status_code == 200:
            return pd.read_csv(pd.compat.StringIO(response.text)), None
        else:
            return None, f"Erreur: {response.status_code}"
    except Exception as e:
        return None, f"Erreur: {str(e)}"

# Fonction pour charger et nettoyer les données
@st.cache_data
def load_and_clean_data(data_source, source_type="file"):
    try:
        if source_type == "file":
            df = pd.read_csv(data_source)
        elif source_type == "drive":
            df, error = download_from_drive(data_source)
            if error:
                st.error(error)
                return None
        else:
            df = data_source

        # Suppression des colonnes inutiles
        df.drop(columns=[col for col in df.columns if "Unnamed" in col], inplace=True, errors='ignore')

        # Renommage des colonnes
        df.rename(columns={
            'Customer ID': 'customer_id',
            'Working Date': 'working_date',
            'Customer Since': 'customer_since'
        }, inplace=True)

        # Conversion des dates
        date_cols = ['created_at', 'working_date', 'customer_since']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # Nettoyage des valeurs manquantes
        df.dropna(subset=['customer_id', 'created_at', 'grand_total'], inplace=True)
        df['category_name_1'].fillna('Unknown', inplace=True)

        # Conversion des types
        df['customer_id'] = df['customer_id'].astype(int)
        return df
    except Exception as e:
        st.error(f"Erreur: {str(e)}")
        return None

# Fonction pour calculer les métriques RFM
@st.cache_data
def calculate_rfm(df, reference_date=None):
    try:
        if reference_date:
            latest_date = pd.to_datetime(reference_date)
        else:
            latest_date = df['created_at'].max()

        rfm_df = df.groupby('customer_id').agg(
            Recency=('created_at', lambda x: (latest_date - x.max()).days),
            Frequency=('increment_id', 'nunique'),
            Monetary=('grand_total', 'sum')
        ).reset_index()
        return rfm_df
    except Exception as e:
        st.error(f"Erreur RFM: {str(e)}")
        return None

# Fonction pour créer les segments RFM
@st.cache_data
def create_rfm_segments(rfm_df):
    rfm_df['R_Score'] = pd.qcut(rfm_df['Recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm_df['F_Score'] = pd.qcut(rfm_df['Frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm_df['M_Score'] = pd.qcut(rfm_df['Monetary'], 5, labels=[1, 2, 3, 4, 5])
    rfm_df['RFM_Score'] = rfm_df['R_Score'].astype(str) + rfm_df['F_Score'].astype(str) + rfm_df['M_Score'].astype(str)

    def segment_customers(row):
        if row['RFM_Score'] in ['555', '554', '544', '545', '454', '455', '445']:
            return 'Champions'
        elif row['RFM_Score'] in ['543', '444', '435', '355', '354', '345', '344', '335']:
            return 'Loyal Customers'
        elif row['RFM_Score'] in ['553', '551', '552', '541', '542', '533', '532', '531', '452', '451']:
            return 'Potential Loyalists'
        elif row['RFM_Score'] in ['512', '511', '422', '421', '412', '411', '311']:
            return 'New Customers'
        elif row['RFM_Score'] in ['244', '334', '343', '353', '143', '234', '243']:
            return 'At Risk'
        elif row['RFM_Score'] in ['155', '254', '144', '214', '215', '115', '114']:
            return 'Can\'t Lose Them'
        elif row['RFM_Score'] in ['111', '112', '121', '131', '141', '151']:
            return 'Lost'
        else:
            return 'Others'

    rfm_df['Segment'] = rfm_df.apply(segment_customers, axis=1)
    return rfm_df

# Fonction pour le clustering DBSCAN
@st.cache_data
def perform_dbscan_clustering(rfm_df, eps=0.5, min_samples=5):
    rfm_vars = ['Recency', 'Frequency', 'Monetary']
    rfm_scaled = StandardScaler().fit_transform(rfm_df[rfm_vars])
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(rfm_scaled)
    rfm_df['Cluster'] = clusters
    if len(set(clusters)) > 1:
        silhouette_avg = silhouette_score(rfm_scaled[clusters != -1], clusters[clusters != -1])
    else:
        silhouette_avg = -1
    return rfm_df, silhouette_avg

# Fonction pour le Market Basket Analysis
@st.cache_data
def perform_market_basket_analysis(df, min_support=0.01):
    basket = df.groupby(['increment_id', 'category_name_1'])['qty_ordered'].sum().unstack().fillna(0)
    basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)
    frequent_itemsets = apriori(basket_sets, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    return frequent_itemsets, rules

# Barre latérale avec menu élégant
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>IAnalyz</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #778899;'>Customer Analytics Dashboard</h3>", unsafe_allow_html=True)
    selected_page = st.selectbox(
        "Navigation",
        ["Accueil", "Import & Nettoyage", "Analyse Exploratoire", "Analyse RFM", "Clustering", "Market Basket Analysis"],
        index=0
    )

# Page Accueil
if selected_page == "Accueil":
    st.markdown("<h1 style='text-align: center;'>Bienvenue sur IAnalyz</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #778899;'>Transformez vos données en insights actionnables</h3>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st_lottie(lottie_animation, height=300, key="analytics")

    st.markdown("---")
    st.markdown("<h2 style='text-align: center;'>Fonctionnalités Clés</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="feature-card">
            <i class="fas fa-chart-line fa-3x" style="color: #4e79a7;"></i>
            <h3>Analyse RFM</h3>
            <p>Segmentez vos clients en fonction de leur comportement d'achat.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="feature-card">
            <i class="fas fa-users fa-3x" style="color: #f28e2b;"></i>
            <h3>Clustering Avancé</h3>
            <p>Identifiez des groupes de clients similaires avec DBSCAN.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="feature-card">
            <i class="fas fa-shopping-basket fa-3x" style="color: #59a14f;"></i>
            <h3>Market Basket</h3>
            <p>Découvrez les associations entre produits pour booster vos ventes.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<h2 style='text-align: center;'>Comment ça marche ?</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='step-card'><h4>1. Import</h4><p>Téléchargez vos données depuis un fichier ou Google Drive.</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='step-card'><h4>2. Analyse</h4><p>Explorez vos données avec des visualisations interactives.</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='step-card'><h4>3. Action</h4><p>Obtenez des recommandations pour optimiser vos campagnes.</p></div>", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("Commencer l'Analyse", key="start_analysis"):
        selected_page = "Import & Nettoyage"
        st.experimental_rerun()

# Page Import & Nettoyage
elif selected_page == "Import & Nettoyage":
    st.markdown("<h1>Import et Nettoyage des Données</h1>", unsafe_allow_html=True)
    import_method = st.radio("Méthode d'import", ["Fichier Local", "Google Drive"])

    if import_method == "Fichier Local":
        uploaded_file = st.file_uploader("Téléchargez votre fichier CSV", type=["csv"])
        if uploaded_file:
            with st.spinner("Chargement et nettoyage en cours..."):
                df = load_and_clean_data(uploaded_file, "file")
                if df is not None:
                    st.session_state.df = df
                    st.success("Données chargées avec succès !")
                    st.markdown(f"<p>Lignes: {len(df):,}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p>Colonnes: {len(df.columns)}</p>", unsafe_allow_html=True)
                    st.dataframe(df.head(10))

                    if st.button("Passer à l'Analyse Exploratoire"):
                        st.experimental_rerun()

    elif import_method == "Google Drive":
        drive_url = st.text_input("URL Google Drive")
        if st.button("Télécharger"):
            with st.spinner("Téléchargement..."):
                df, error = download_from_drive(drive_url)
                if df is not None:
                    st.session_state.df = df
                    st.success("Données téléchargées avec succès !")
                    st.dataframe(df.head(10))

# Page Analyse Exploratoire
elif selected_page == "Analyse Exploratoire":
    if 'df' not in st.session_state:
        st.warning("Veuillez d'abord importer vos données.")
    else:
        df = st.session_state.df
        st.markdown("<h1>Analyse Exploratoire</h1>", unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"<div class='metric-card'><h4>Lignes</h4><p>{len(df):,}</p></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='metric-card'><h4>Colonnes</h4><p>{len(df.columns)}</p></div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='metric-card'><h4>Clients</h4><p>{df['customer_id'].nunique():,}</p></div>", unsafe_allow_html=True)
        with col4:
            st.markdown(f"<div class='metric-card'><h4>Commandes</h4><p>{df['increment_id'].nunique():,}</p></div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("<h2>Top Catégories</h2>", unsafe_allow_html=True)
        top_categories = df['category_name_1'].value_counts().head(10)
        fig = px.bar(top_categories, title="Top 10 Catégories", color_discrete_sequence=['#4e79a7'])
        st.plotly_chart(fig, use_container_width=True)

        if st.button("Analyse RFM"):
            st.session_state.rfm_df = calculate_rfm(df)
            st.experimental_rerun()

# Page Analyse RFM
elif selected_page == "Analyse RFM":
    if 'rfm_df' not in st.session_state:
        st.warning("Veuillez d'abord calculer les métriques RFM.")
    else:
        rfm_df = st.session_state.rfm_df
        st.markdown("<h1>Analyse RFM</h1>", unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"<div class='metric-card'><h4>Récence Moyenne</h4><p>{rfm_df['Recency'].mean():.1f} jours</p></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='metric-card'><h4>Fréquence Moyenne</h4><p>{rfm_df['Frequency'].mean():.1f}</p></div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='metric-card'><h4>Valeur Moyenne</h4><p>${rfm_df['Monetary'].mean():.2f}</p></div>", unsafe_allow_html=True)
        with col4:
            st.markdown(f"<div class='metric-card'><h4>Segments</h4><p>{rfm_df['Segment'].nunique()}</p></div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("<h2>Répartition des Segments</h2>", unsafe_allow_html=True)
        segment_counts = rfm_df['Segment'].value_counts()
        fig = px.pie(segment_counts, values=segment_counts.values, names=segment_counts.index, title="Segments Clients")
        st.plotly_chart(fig, use_container_width=True)

        if st.button("Clustering DBSCAN"):
            rfm_clustered, silhouette_avg = perform_dbscan_clustering(rfm_df)
            st.session_state.rfm_clustered = rfm_clustered
            st.session_state.silhouette_avg = silhouette_avg
            st.experimental_rerun()

# Page Clustering
elif selected_page == "Clustering":
    if 'rfm_clustered' not in st.session_state:
        st.warning("Veuillez d'abord effectuer le clustering.")
    else:
        rfm_clustered = st.session_state.rfm_clustered
        silhouette_avg = st.session_state.silhouette_avg
        st.markdown("<h1>Clustering DBSCAN</h1>", unsafe_allow_html=True)

        st.markdown(f"<div class='metric-card'><h4>Score de Silhouette</h4><p>{silhouette_avg:.3f}</p></div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("<h2>Visualisation 3D des Clusters</h2>", unsafe_allow_html=True)
        fig = px.scatter_3d(
            rfm_clustered,
            x='Recency',
            y='Frequency',
            z='Monetary',
            color='Cluster',
            title="Clusters DBSCAN"
        )
        st.plotly_chart(fig, use_container_width=True)

# Page Market Basket Analysis
elif selected_page == "Market Basket Analysis":
    if 'df' not in st.session_state:
        st.warning("Veuillez d'abord importer vos données.")
    else:
        df = st.session_state.df
        st.markdown("<h1>Market Basket Analysis</h1>", unsafe_allow_html=True)

        min_support = st.slider("Support Minimum", 0.001, 0.1, 0.01)
        frequent_itemsets, rules = perform_market_basket_analysis(df, min_support)

        st.markdown("<h2>Règles d'Association</h2>", unsafe_allow_html=True)
        st.dataframe(rules.head(10))

        st.markdown("---")
        st.markdown("<h2>Heatmap des Co-occurrences</h2>", unsafe_allow_html=True)
        basket = df.groupby(['increment_id', 'category_name_1'])['qty_ordered'].sum().unstack().fillna(0)
        cooccurrence_matrix = basket.T.dot(basket)
        fig = px.imshow(cooccurrence_matrix, labels=dict(x="Catégories", y="Catégories"), title="Co-occurrences")
        st.plotly_chart(fig, use_container_width=True)
