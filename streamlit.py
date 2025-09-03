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
    page_title="E-commerce Analytics Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre principal
st.title("🛒 E-commerce Analytics Dashboard")
st.markdown("---")

# Sidebar pour la navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choisissez une section",
    ["📤 Upload & Nettoyage", "📊 Analyse RFM", "🎯 Clustering", "🛍️ Market Basket Analysis"]
)

# Fonctions utilitaires
@st.cache_data
def load_and_clean_data(uploaded_file):
    """Charge et nettoie les données"""
    try:
        # Lecture du fichier CSV
        df = pd.read_csv(uploaded_file, low_memory=False)
        
        # Nettoyage des colonnes inutiles
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
        
        # Mapping des méthodes de paiement
        payment_method_mapping = {
            'cod': 'direct', 'ublcreditcard': 'credit_card', 'mygateway': 'credit_card',
            'customercredit': 'direct', 'cashatdoorstep': 'direct', 'mcblite': 'mobile-money',
            'internetbanking': 'direct', 'marketingexpense': 'direct', 'productcredit': 'direct',
            'financesettlement': 'direct', 'Payaxis': 'credit_card', 'jazzvoucher': 'mobile-money',
            'jazzwallet': 'mobile-money', 'Easypay': 'mobile-money', 'Easypay_MA': 'mobile-money',
            'easypay_voucher': 'mobile-money', 'bankalfalah': 'direct', 'apg': 'credit_card',
            'Unknown': 'direct'
        }
        df['payment_method'] = df['payment_method'].map(payment_method_mapping).fillna('direct')
        
        # Nettoyage des données essentielles
        essential_cols = ['customer_id', 'created_at', 'grand_total', 'status']
        if 'category_name_1' in df.columns:
            essential_cols.append('category_name_1')
        
        df = df.dropna(subset=[col for col in essential_cols if col in df.columns])
        
        # Gestion des valeurs manquantes
        df['discount_amount'] = df['discount_amount'].fillna(0)
        df['sales_commission_code'] = df['sales_commission_code'].fillna('Unknown')
        
        # Conversion des dates
        date_cols = ['created_at', 'customer_since', 'working_date']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Traitement des outliers
        numeric_cols = ['price', 'qty_ordered', 'grand_total', 'discount_amount']
        for col in numeric_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[col] = np.clip(df[col], lower_bound, upper_bound)
        
        return df
        
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
        return None

@st.cache_data
def calculate_rfm(df):
    """Calcule les métriques RFM"""
    try:
        df_filtered = df.dropna(subset=['created_at'])
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

@st.cache_data
def perform_clustering(rfm_df, n_clusters=4):
    """Effectue le clustering K-means"""
    # Préparation des données
    rfm_vars = ['Recency', 'Frequency', 'Monetary']
    rfm_transformed = rfm_df.copy()
    
    # Transformation log
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

def parse_items(cell):
    """Parse les items d'une cellule"""
    if pd.isna(cell):
        return []
    parts = [x.strip().strip('"').strip("'") for x in str(cell).split(",")]
    parts = [re.sub(r"\s+", " ", p).strip() for p in parts if p and p.strip()]
    return parts

@st.cache_data
def prepare_market_basket_data(df, sample_size=50000):
    """Prépare les données pour l'analyse Market Basket"""
    # Échantillonnage pour les performances
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df.copy()
    
    transactions = defaultdict(set)
    
    if 'items' in df_sample.columns:
        grouped = df_sample.groupby("increment_id")["items"].apply(list)
        for tid, items_list in grouped.items():
            for raw in items_list:
                for item in parse_items(raw):
                    transactions[tid].add(item)
    elif 'category_name_1' in df_sample.columns:
        # Utiliser les catégories comme items
        grouped = df_sample.groupby("increment_id")["category_name_1"].apply(list)
        for tid, categories in grouped.items():
            for cat in categories:
                if pd.notna(cat):
                    transactions[tid].add(str(cat))
    
    # Convertir en format baskets
    baskets = []
    for tid, items in transactions.items():
        if len(items) > 1:  # Seulement les transactions avec plus d'un item
            baskets.append(sorted(list(items)))
    
    return baskets, transactions

def simple_apriori(baskets, min_support=0.01, min_confidence=0.5):
    """Implémentation simple d'Apriori"""
    # Compter les items individuels
    item_counts = defaultdict(int)
    for basket in baskets:
        for item in basket:
            item_counts[item] += 1
    
    total_baskets = len(baskets)
    min_support_count = int(min_support * total_baskets)
    
    # Items fréquents
    frequent_items = {item: count for item, count in item_counts.items() 
                     if count >= min_support_count}
    
    # Générer des paires
    rules = []
    for basket in baskets:
        basket_items = [item for item in basket if item in frequent_items]
        for i, item_a in enumerate(basket_items):
            for item_b in basket_items[i+1:]:
                rules.append((item_a, item_b, frequent_items[item_a], frequent_items[item_b]))
    
    # Calculer support et confiance pour les paires
    pair_counts = defaultdict(int)
    for rule in rules:
        pair_counts[(rule[0], rule[1])] += 1
    
    association_rules = []
    for (antecedent, consequent), support_count in pair_counts.items():
        if support_count >= min_support_count:
            support = support_count / total_baskets
            confidence = support_count / frequent_items[antecedent]
            
            if confidence >= min_confidence:
                association_rules.append({
                    'Antecedent': antecedent,
                    'Consequent': consequent,
                    'Support': support,
                    'Confidence': confidence,
                    'Lift': confidence / (frequent_items[consequent] / total_baskets)
                })
    
    return association_rules, frequent_items

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
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Nombre de lignes", len(df))
            with col2:
                st.metric("Nombre de colonnes", len(df.columns))
            with col3:
                st.metric("Clients uniques", df['customer_id'].nunique())
            
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

elif page == "📊 Analyse RFM":
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

elif page == "🛍️ Market Basket Analysis":
    if 'df' not in st.session_state:
        st.warning("⚠️ Veuillez d'abord charger les données dans la section 'Upload & Nettoyage'")
    else:
        st.header("🛍️ Market Basket Analysis")
        
        df = st.session_state.df
        
        # Paramètres
        col1, col2 = st.columns(2)
        with col1:
            min_support = st.slider("Support minimum", 0.001, 0.1, 0.01, 0.001)
        with col2:
            min_confidence = st.slider("Confiance minimum", 0.1, 1.0, 0.5, 0.1)
        
        sample_size = st.sidebar.slider("Taille de l'échantillon", 1000, 100000, 50000, 1000)
        
        with st.spinner("Analyse des associations..."):
            baskets, transactions = prepare_market_basket_data(df, sample_size)
        
        if baskets:
            st.success(f"✅ {len(baskets)} paniers analysés")
            
            # Calcul des règles d'association
            with st.spinner("Génération des règles d'association..."):
                rules, frequent_items = simple_apriori(baskets, min_support, min_confidence)
            
            if rules:
                st.success(f"✅ {len(rules)} règles d'association trouvées")
                
                # Conversion en DataFrame
                rules_df = pd.DataFrame(rules)
                rules_df = rules_df.sort_values('Lift', ascending=False)
                
                # Métriques
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Règles trouvées", len(rules_df))
                with col2:
                    st.metric("Items fréquents", len(frequent_items))
                with col3:
                    st.metric("Lift maximum", f"{rules_df['Lift'].max():.2f}")
                
                # Tableau des règles
                st.subheader("Règles d'association")
                st.dataframe(rules_df.head(20))
                
                # Visualisations
                col1, col2 = st.columns(2)
                
                with col1:
                    # Graphique Support vs Confiance
                    fig = px.scatter(rules_df, x='Support', y='Confidence', 
                                   size='Lift', hover_data=['Antecedent', 'Consequent'],
                                   title="Support vs Confiance")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Top 10 règles par Lift
                    top_rules = rules_df.head(10)
                    fig = px.bar(top_rules, x='Lift', y='Antecedent',
                               orientation='h', title="Top 10 règles par Lift")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Items les plus fréquents
                st.subheader("Items les plus fréquents")
                frequent_items_df = pd.DataFrame(list(frequent_items.items()), 
                                               columns=['Item', 'Fréquence'])
                frequent_items_df = frequent_items_df.sort_values('Fréquence', ascending=False)
                
                fig = px.bar(frequent_items_df.head(20), x='Fréquence', y='Item',
                           orientation='h', title="Top 20 items les plus fréquents")
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.warning("Aucune règle d'association trouvée avec les paramètres actuels. Essayez de réduire le support ou la confiance minimum.")
        else:
            st.warning("Aucun panier trouvé dans les données. Vérifiez que les colonnes 'items' ou 'category_name_1' existent.")

# Footer
st.markdown("---")
st.markdown("Application développée avec Streamlit - Analyse complète des données e-commerce")