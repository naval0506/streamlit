import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
import re
from collections import defaultdict
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="E-commerce Pakistan Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre principal
st.title("🛒 Pakistan E-commerce Analytics Dashboard")
st.markdown("Analyse du plus grand dataset e-commerce du Pakistan")
st.markdown("---")

# Sidebar pour la navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choisissez une section",
    ["📤 Upload & Nettoyage", "📊 Analyse Exploratoire", "📈 Analyse RFM", "🎯 Clustering", "🛍️ Market Basket Analysis"]
)

# Fonctions utilitaires adaptées au dataset
@st.cache_data
def load_and_clean_data(uploaded_file):
    """Charge et nettoie les données spécifiques au dataset Pakistan"""
    try:
        # Lecture du fichier CSV
        df = pd.read_csv(uploaded_file, low_memory=False)
        
        # Nettoyage des colonnes inutiles (100% valeurs manquantes)
        cols_to_drop = ['Unnamed: 21', 'Unnamed: 22', 'Unnamed: 23', 'Unnamed: 24', 'Unnamed: 25']
        cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)
        
        # Renommage des colonnes
        df.rename(columns={
            'Customer ID': 'customer_id', 
            'Working Date': 'working_date', 
            'BI Status': 'bi_status',
            ' MV ': 'mv', 
            'Year': 'year', 
            'Month': 'month', 
            'Customer Since': 'customer_since',
            'M-Y': 'm_y', 
            'FY': 'fy'
        }, inplace=True)
        
        # Mapping des méthodes de paiement (adapté au dataset Pakistan)
        payment_method_mapping = {
            'cod': 'autres', 'ublcreditcard': 'carte_bancaire', 'mygateway': 'carte_bancaire',
            'customercredit': 'autres', 'cashatdoorstep': 'autres', 'mcblite': 'autres',
            'internetbanking': 'autres', 'marketingexpense': 'autres', 'productcredit': 'autres',
            'financesettlement': 'autres', 'Payaxis': 'carte_bancaire', 'jazzvoucher': 'autres',
            'jazzwallet': 'autres', 'Easypay': 'autres', 'Easypay_MA': 'autres',
            'easypay_voucher': 'autres', 'bankalfalah': 'autres', 'apg': 'carte_bancaire',
            'Unknown': 'autres'
        }
        df['payment_method'] = df['payment_method'].map(payment_method_mapping).fillna('autres')
        
        # Nettoyage des données essentielles - gestion des 44% de valeurs manquantes
        essential_cols = ['customer_id', 'created_at', 'grand_total', 'status', 'category_name_1']
        df = df.dropna(subset=[col for col in essential_cols if col in df.columns])
        
        # Gestion des valeurs manquantes pour les autres colonnes
        numeric_cols = ['price', 'qty_ordered', 'discount_amount', 'year', 'month']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Conversion des dates avec gestion des erreurs
        date_cols = ['created_at', 'customer_since', 'working_date']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Traitement des outliers pour les valeurs numériques
        numeric_cols = ['price', 'qty_ordered', 'grand_total', 'discount_amount']
        for col in numeric_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[col] = np.clip(df[col], lower_bound, upper_bound)
        
        # Suppression des doublons (44% dans le dataset original)
        initial_count = len(df)
        df.drop_duplicates(inplace=True)
        final_count = len(df)
        st.info(f"Suppression de {initial_count - final_count} doublons ({((initial_count - final_count)/initial_count*100):.2f}%)")
        
        return df
        
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
        return None

@st.cache_data
def calculate_rfm(df):
    """Calcule les métriques RFM adaptées au dataset"""
    try:
        df_filtered = df.dropna(subset=['created_at', 'customer_id', 'grand_total'])
        
        # Vérification des données nécessaires
        if df_filtered.empty:
            st.error("Données insuffisantes pour le calcul RFM")
            return None
            
        latest_date = df_filtered['created_at'].max()
        
        # Calcul de la récence
        recency_df = df_filtered.groupby('customer_id')['created_at'].max().reset_index()
        recency_df['Recency'] = (latest_date - recency_df['created_at']).dt.days
        
        # Calcul de la fréquence et du montant
        rfm_df = df_filtered.groupby('customer_id').agg(
            Frequency=('increment_id', 'nunique'),
            Monetary=('grand_total', 'sum')
        ).reset_index()
        
        # Fusion des données
        rfm_df = rfm_df.merge(recency_df[['customer_id', 'Recency']], on='customer_id')
        rfm_df = rfm_df[['customer_id', 'Recency', 'Frequency', 'Monetary']]
        
        return rfm_df
        
    except Exception as e:
        st.error(f"Erreur lors du calcul RFM: {e}")
        return None

def create_rfm_segments(rfm_df):
    """Crée les segments RFM"""
    try:
        # Calcul des quintiles
        rfm_df['R_Score'] = pd.qcut(rfm_df['Recency'], 5, labels=[5, 4, 3, 2, 1])
        rfm_df['F_Score'] = pd.qcut(rfm_df['Frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
        rfm_df['M_Score'] = pd.qcut(rfm_df['Monetary'], 5, labels=[1, 2, 3, 4, 5])
        
        # Score RFM combiné
        rfm_df['RFM_Score'] = rfm_df['R_Score'].astype(str) + rfm_df['F_Score'].astype(str) + rfm_df['M_Score'].astype(str)
        
        # Segmentation
        def segment_customers(row):
            if row['RFM_Score'] in ['555', '554', '544', '545', '454', '455', '445']:
                return 'Champions'
            elif row['RFM_Score'] in ['543', '444', '435', '355', '354', '345', '344', '335']:
                return 'Loyal Customers'
            elif row['RFM_Score'] in ['553', '551', '552', '541', '542', '533', '532', '531', '452', '451']:
                return 'Potential Loyalists'
            elif row['RFM_Score'] in ['512', '511', '422', '421', '412', '411', '311']:
                return 'New Customers'
            elif row['RFM_Score'] in ['155', '154', '144', '214', '215', '115', '114']:
                return 'At Risk'
            elif row['RFM_Score'] in ['155', '254', '334', '343', '353', '143', '244']:
                return 'Cannot Lose Them'
            elif row['RFM_Score'] in ['111', '112', '121', '131', '141', '151']:
                return 'Lost'
            else:
                return 'Others'
        
        rfm_df['Segment'] = rfm_df.apply(segment_customers, axis=1)
        return rfm_df
    except Exception as e:
        st.error(f"Erreur lors de la création des segments RFM: {e}")
        return None

@st.cache_data
def perform_clustering(rfm_df, n_clusters=4):
    """Effectue le clustering K-means"""
    try:
        # Préparation des données
        rfm_vars = ['Recency', 'Frequency', 'Monetary']
        rfm_transformed = rfm_df.copy()
        
        # Transformation log pour gérer la skewness
        for col in rfm_vars:
            rfm_transformed[col] = np.log1p(rfm_transformed[col])
        
        # Normalisation
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm_transformed[rfm_vars])
        
        # Clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(rfm_scaled)
        
        rfm_df['Cluster'] = clusters
        silhouette_avg = silhouette_score(rfm_scaled, clusters)
        
        return rfm_df, rfm_scaled, silhouette_avg, scaler, kmeans
    except Exception as e:
        st.error(f"Erreur lors du clustering: {e}")
        return None, None, None, None, None

def analyze_payment_methods(df):
    """Analyse des méthodes de paiement"""
    if 'payment_method' not in df.columns:
        return None
        
    payment_counts = df['payment_method'].value_counts()
    fig = px.pie(values=payment_counts.values, names=payment_counts.index,
                 title="Répartition des méthodes de paiement")
    return fig

def analyze_sales_trends(df):
    """Analyse des tendances de ventes"""
    if 'created_at' not in df.columns:
        return None
        
    df['order_month'] = df['created_at'].dt.to_period('M').dt.to_timestamp()
    monthly_sales = df.groupby('order_month').agg({
        'grand_total': 'sum',
        'increment_id': 'nunique'
    }).reset_index()
    
    monthly_sales.columns = ['Mois', 'Chiffre d\'affaires', 'Nombre de commandes']
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=monthly_sales['Mois'], y=monthly_sales['Chiffre d\'affaires'], 
                  name="Chiffre d'affaires", line=dict(color='blue')),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(x=monthly_sales['Mois'], y=monthly_sales['Nombre de commandes'], 
                  name="Nombre de commandes", line=dict(color='red')),
        secondary_y=True,
    )
    
    fig.update_layout(title_text="Évolution des ventes mensuelles")
    fig.update_xaxes(title_text="Mois")
    fig.update_yaxes(title_text="Chiffre d'affaires", secondary_y=False)
    fig.update_yaxes(title_text="Nombre de commandes", secondary_y=True)
    
    return fig

def analyze_categories(df):
    """Analyse des catégories de produits"""
    if 'category_name_1' not in df.columns:
        return None
        
    category_stats = df.groupby('category_name_1').agg({
        'grand_total': 'sum',
        'increment_id': 'nunique',
        'qty_ordered': 'sum'
    }).nlargest(10, 'grand_total')
    
    category_stats.columns = ['Chiffre d\'affaires', 'Nombre de commandes', 'Quantité vendue']
    
    fig = px.bar(category_stats, x=category_stats.index, y='Chiffre d\'affaires',
                 title="Top 10 catégories par chiffre d'affaires")
    return fig

# Interface principale
if page == "📤 Upload & Nettoyage":
    st.header("📤 Upload et Nettoyage des Données")
    
    uploaded_file = st.file_uploader("Téléchargez votre fichier CSV", type=["csv"])
    
    if uploaded_file is not None:
        with st.spinner("Chargement et nettoyage des données..."):
            df = load_and_clean_data(uploaded_file)
        
        if df is not None:
            st.success(f"✅ Données chargées avec succès! Shape: {df.shape}")
            
            # Stocker les données dans session state
            st.session_state.df = df
            
            # Affichage des informations générales
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Nombre de lignes", len(df))
            with col2:
                st.metric("Nombre de colonnes", len(df.columns))
            with col3:
                st.metric("Clients uniques", df['customer_id'].nunique())
            with col4:
                st.metric("Commandes uniques", df['increment_id'].nunique())
            
            # Aperçu des données
            st.subheader("Aperçu des données")
            st.dataframe(df.head())
            
            # Informations sur les colonnes
            st.subheader("Informations sur les colonnes")
            col_info = pd.DataFrame({
                'Type': df.dtypes,
                'Non-null Count': df.count(),
                'Null Count': df.isnull().sum(),
                'Null %': (df.isnull().sum() / len(df) * 100).round(2)
            })
            st.dataframe(col_info)
            
            # Statistiques descriptives
            st.subheader("Statistiques descriptives")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.dataframe(df[numeric_cols].describe())

elif page == "📊 Analyse Exploratoire":
    if 'df' not in st.session_state:
        st.warning("⚠️ Veuillez d'abord charger les données dans la section 'Upload & Nettoyage'")
    else:
        st.header("📊 Analyse Exploratoire des Données")
        
        df = st.session_state.df
        
        # Métriques clés
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_sales = df['grand_total'].sum()
            st.metric("Chiffre d'affaires total", f"${total_sales:,.2f}")
        with col2:
            avg_order_value = df['grand_total'].mean()
            st.metric("Panier moyen", f"${avg_order_value:.2f}")
        with col3:
            total_orders = df['increment_id'].nunique()
            st.metric("Nombre total de commandes", total_orders)
        with col4:
            total_customers = df['customer_id'].nunique()
            st.metric("Nombre total de clients", total_customers)
        
        # Analyse des méthodes de paiement
        st.subheader("Méthodes de paiement")
        payment_fig = analyze_payment_methods(df)
        if payment_fig:
            st.plotly_chart(payment_fig, use_container_width=True)
        else:
            st.warning("Données de paiement non disponibles")
        
        # Tendances des ventes
        st.subheader("Tendances des ventes")
        sales_fig = analyze_sales_trends(df)
        if sales_fig:
            st.plotly_chart(sales_fig, use_container_width=True)
        else:
            st.warning("Données de date non disponibles pour l'analyse des tendances")
        
        # Analyse des catégories
        st.subheader("Top catégories")
        category_fig = analyze_categories(df)
        if category_fig:
            st.plotly_chart(category_fig, use_container_width=True)
        else:
            st.warning("Données de catégories non disponibles")
        
        # Distribution des montants de commande
        st.subheader("Distribution des montants de commande")
        if 'grand_total' in df.columns:
            fig = px.histogram(df, x='grand_total', nbins=50, 
                             title="Distribution des montants de commande",
                             labels={'grand_total': 'Montant de commande'})
            st.plotly_chart(fig, use_container_width=True)

elif page == "📈 Analyse RFM":
    if 'df' not in st.session_state:
        st.warning("⚠️ Veuillez d'abord charger les données dans la section 'Upload & Nettoyage'")
    else:
        st.header("📊 Analyse RFM (Recency, Frequency, Monetary)")
        
        df = st.session_state.df
        
        with st.spinner("Calcul des métriques RFM..."):
            rfm_df = calculate_rfm(df)
        
        if rfm_df is not None:
            # Créer les segments
            rfm_segments = create_rfm_segments(rfm_df)
            st.session_state.rfm_df = rfm_segments
            
            # Métriques RFM
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Récence moyenne", f"{rfm_df['Recency'].mean():.1f} jours")
            with col2:
                st.metric("Fréquence moyenne", f"{rfm_df['Frequency'].mean():.1f}")
            with col3:
                st.metric("Valeur monétaire moyenne", f"${rfm_df['Monetary'].mean():.2f}")
            with col4:
                st.metric("Segments identifiés", rfm_segments['Segment'].nunique())
            
            # Distribution des segments
            st.subheader("Distribution des segments clients")
            segment_counts = rfm_segments['Segment'].value_counts()
            
            fig = px.pie(values=segment_counts.values, names=segment_counts.index,
                        title="Répartition des segments clients")
            st.plotly_chart(fig, use_container_width=True)
            
            # Tableau des segments
            st.subheader("Statistiques par segment")
            segment_stats = rfm_segments.groupby('Segment').agg({
                'Recency': 'mean',
                'Frequency': 'mean',
                'Monetary': 'mean',
                'customer_id': 'count'
            }).round(2)
            segment_stats.columns = ['Récence moyenne', 'Fréquence moyenne', 'Valeur moyenne', 'Nombre de clients']
            st.dataframe(segment_stats)
            
            # Graphiques RFM
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.scatter(rfm_segments, x='Frequency', y='Monetary', 
                               color='Segment', size='Recency',
                               title="Fréquence vs Valeur Monétaire")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.scatter(rfm_segments, x='Recency', y='Frequency', 
                               color='Segment', size='Monetary',
                               title="Récence vs Fréquence")
                st.plotly_chart(fig, use_container_width=True)
            
            # Graphique 3D
            st.subheader("Visualisation 3D des segments RFM")
            fig = px.scatter_3d(rfm_segments, x='Recency', y='Frequency', z='Monetary',
                              color='Segment', title="Segments RFM en 3D")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Impossible de calculer les métriques RFM. Vérifiez que les colonnes nécessaires existent.")

elif page == "🎯 Clustering":
    if 'rfm_df' not in st.session_state:
        st.warning("⚠️ Veuillez d'abord effectuer l'analyse RFM")
    else:
        st.header("🎯 Clustering des Clients")
        
        rfm_df = st.session_state.rfm_df
        
        # Paramètres du clustering
        n_clusters = st.sidebar.slider("Nombre de clusters", min_value=2, max_value=10, value=4)
        
        with st.spinner("Clustering en cours..."):
            rfm_clustered, rfm_scaled, silhouette_avg, scaler, kmeans = perform_clustering(rfm_df, n_clusters)
        
        if rfm_clustered is not None:
            st.session_state.rfm_clustered = rfm_clustered
            
            # Métriques du clustering
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Score de silhouette", f"{silhouette_avg:.3f}")
            with col2:
                st.metric("Nombre de clusters", n_clusters)
            
            # Distribution des clusters
            cluster_counts = rfm_clustered['Cluster'].value_counts().sort_index()
            fig = px.bar(x=cluster_counts.index, y=cluster_counts.values,
                        title="Distribution des clusters",
                        labels={'x': 'Cluster', 'y': 'Nombre de clients'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistiques par cluster
            st.subheader("Caractéristiques des clusters")
            cluster_stats = rfm_clustered.groupby('Cluster').agg({
                'Recency': 'mean',
                'Frequency': 'mean',
                'Monetary': 'mean',
                'customer_id': 'count'
            }).round(2)
            cluster_stats.columns = ['Récence moyenne', 'Fréquence moyenne', 'Valeur moyenne', 'Nombre de clients']
            st.dataframe(cluster_stats)
            
            # Visualisations des clusters
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.scatter(rfm_clustered, x='Frequency', y='Monetary',
                               color='Cluster', title="Clusters: Fréquence vs Valeur")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.scatter(rfm_clustered, x='Recency', y='Frequency',
                               color='Cluster', title="Clusters: Récence vs Fréquence")
                st.plotly_chart(fig, use_container_width=True)
            
            # Graphique 3D des clusters
            st.subheader("Visualisation 3D des clusters")
            fig = px.scatter_3d(rfm_clustered, x='Recency', y='Frequency', z='Monetary',
                              color='Cluster', title="Clusters en 3D")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Erreur lors du clustering des données.")

elif page == "🛍️ Market Basket Analysis":
    st.header("🛍️ Market Basket Analysis")
    st.info("Cette fonctionnalité nécessite des données de produits détaillées. Veuillez vérifier que votre dataset contient des informations sur les produits ou catégories.")
    
    if 'df' not in st.session_state:
        st.warning("⚠️ Veuillez d'abord charger les données dans la section 'Upload & Nettoyage'")
    else:
        df = st.session_state.df
        
        # Vérification des colonnes nécessaires
        if 'category_name_1' not in df.columns and 'sku' not in df.columns:
            st.error("Colonnes nécessaires pour l'analyse Market Basket non trouvées (category_name_1 ou sku)")
        else:
            st.success("Colonnes nécessaires détectées. L'analyse Market Basket peut être implémentée ici.")
            
            # Afficher les top catégories
            if 'category_name_1' in df.columns:
                top_categories = df['category_name_1'].value_counts().head(10)
                fig = px.bar(x=top_categories.values, y=top_categories.index, 
                           orientation='h', title="Top 10 Catégories de Produits")
                st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**E-commerce Pakistan Analytics Dashboard** - Analyse des données de vente au détail")
st.markdown("*Dataset: Pakistan's Largest E-commerce Dataset*")