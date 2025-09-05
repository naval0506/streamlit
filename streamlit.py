import  streamlit as st
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
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

# CSS personnalisé
st.markdown("""
<style>
.feature-card {
    background: white;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    padding: 20px;
    text-align: center;
    margin: 10px;
    transition: transform 0.3s;
}

.feature-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
}

.step-card {
    background: white;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    padding: 20px;
    text-align: center;
    margin: 10px;
}

.metric-card {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    border-radius: 10px;
    padding: 15px;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    margin: 10px;
}

.metric-card h4 {
    color: #555;
    margin-bottom: 5px;
}

.metric-card p {
    font-size: 1.5em;
    font-weight: bold;
    color: #333;
}

.stButton>button {
    background-color: #4e79a7;
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 5px;
    cursor: pointer;
}

.stButton>button:hover {
    background-color: #3a5f8a;
}
</style>
""", unsafe_allow_html=True)

# Chargement d'une animation Lottie
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

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

        # Vérification des colonnes essentielles
        required_cols = ['customer_id', 'created_at', 'grand_total', 'increment_id', 'qty_ordered']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.warning(f"Colonnes manquantes: {missing_cols}")
            # Créer des colonnes manquantes avec des valeurs par défaut
            for col in missing_cols:
                if col == 'customer_id':
                    df['customer_id'] = range(1, len(df) + 1)
                elif col == 'created_at':
                    df['created_at'] = pd.date_range(start='2023-01-01', periods=len(df), freq='D')
                elif col == 'grand_total':
                    df['grand_total'] = np.random.uniform(10, 500, len(df))
                elif col == 'increment_id':
                    df['increment_id'] = [f'ORD_{i:06d}' for i in range(1, len(df) + 1)]
                elif col == 'qty_ordered':
                    df['qty_ordered'] = np.random.randint(1, 10, len(df))

        # Ajouter category_name_1 si manquant
        if 'category_name_1' not in df.columns:
            categories = ['Electronics', 'Clothing', 'Home', 'Books', 'Sports', 'Beauty', 'Food', 'Unknown']
            df['category_name_1'] = np.random.choice(categories, len(df))

        # Renommage des colonnes
        column_mapping = {
            'Customer ID': 'customer_id',
            'Working Date': 'working_date',
            'Customer Since': 'customer_since',
            'Created At': 'created_at',
            'Grand Total': 'grand_total',
            'Increment ID': 'increment_id',
            'Qty Ordered': 'qty_ordered',
            'Category Name': 'category_name_1'
        }
        df.rename(columns=column_mapping, inplace=True)

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

# Barre latérale avec menu
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>IAnalyz</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #778899;'>Customer Analytics Dashboard</h3>", unsafe_allow_html=True)
    
    selected_page = option_menu(
        menu_title=None,
        options=["Accueil", "Import & Nettoyage", "Analyse Exploratoire", "Analyse RFM", "Clustering", "Market Basket Analysis"],
        icons=["house", "cloud-upload", "graph-up", "people", "diagram-3", "basket"],
        default_index=0,
        styles={
            "container": {"padding": "0!important"},
            "icon": {"color": "orange", "font-size": "18px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px"},
            "nav-link-selected": {"background-color": "#4e79a7"},
        }
    )
    
    st.markdown("---")
    st.markdown("**IAnalyz v1.0**")
    st.markdown("Développé avec Streamlit")

# Animation Lottie pour l'accueil
lottie_url = "https://assets9.lottiefiles.com/packages/lf20_0yfsb3a5.json"
lottie_animation = load_lottieurl(lottie_url)

# Page Accueil
if selected_page == "Accueil":
    st.markdown("<h1 style='text-align: center;'>Bienvenue sur IAnalyz</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #778899;'>Transformez vos données en insights actionnables</h3>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if lottie_animation:
            st_lottie(lottie_animation, height=300, key="analytics")
        else:
            st.image("https://via.placeholder.com/600x300?text=IAnalytics+Dashboard", use_column_width=True)

    st.markdown("---")
    st.markdown("<h2 style='text-align: center;'>Fonctionnalités Clés</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>📊 Analyse RFM</h3>
            <p>Segmentez vos clients en fonction de leur comportement d'achat.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>👥 Clustering Avancé</h3>
            <p>Identifiez des groupes de clients similaires avec DBSCAN.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>🛒 Market Basket</h3>
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
        st.session_state.page = "Import & Nettoyage"
        st.experimental_rerun()

# Page Import & Nettoyage
elif selected_page == "Import & Nettoyage":
    st.markdown("<h1>Import et Nettoyage des Données</h1>", unsafe_allow_html=True)
    import_method = st.radio("Méthode d'import", ["Fichier Local", "Google Drive", "Données Exemple"])

    if import_method == "Fichier Local":
        uploaded_file = st.file_uploader("Téléchargez votre fichier CSV", type=["csv"])
        if uploaded_file:
            with st.spinner("Chargement et nettoyage en cours..."):
                df = load_and_clean_data(uploaded_file, "file")
                if df is not None:
                    st.session_state.df = df
                    st.success("✅ Données chargées avec succès !")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"<div class='metric-card'><h4>Lignes</h4><p>{len(df):,}</p></div>", unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"<div class='metric-card'><h4>Colonnes</h4><p>{len(df.columns)}</p></div>", unsafe_allow_html=True)
                    
                    st.dataframe(df.head(10))
                    
                    if st.button("Passer à l'Analyse Exploratoire"):
                        st.experimental_rerun()

    elif import_method == "Google Drive":
        drive_url = st.text_input("URL Google Drive", placeholder="https://drive.google.com/...")
        if st.button("Télécharger depuis Drive"):
            with st.spinner("Téléchargement..."):
                df, error = download_from_drive(drive_url)
                if df is not None:
                    df = load_and_clean_data(df, "dataframe")
                    st.session_state.df = df
                    st.success("✅ Données téléchargées avec succès !")
                    st.dataframe(df.head(10))

    elif import_method == "Données Exemple":
        if st.button("Charger des données d'exemple"):
            # Créer des données d'exemple
            np.random.seed(42)
            n_rows = 1000
            dates = pd.date_range('2023-01-01', '2023-12-31', n_rows)
            
            example_data = {
                'customer_id': np.random.randint(1, 101, n_rows),
                'created_at': dates,
                'grand_total': np.random.uniform(10, 500, n_rows),
                'increment_id': [f'ORD_{i:06d}' for i in range(1, n_rows + 1)],
                'qty_ordered': np.random.randint(1, 5, n_rows),
                'category_name_1': np.random.choice(['Electronics', 'Clothing', 'Home', 'Books', 'Sports'], n_rows)
            }
            
            df = pd.DataFrame(example_data)
            st.session_state.df = df
            st.success("✅ Données d'exemple chargées !")
            st.dataframe(df.head(10))

# Page Analyse Exploratoire
elif selected_page == "Analyse Exploratoire":
    if 'df' not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ Veuillez d'abord importer vos données.")
        if st.button("Aller à l'import"):
            st.session_state.page = "Import & Nettoyage"
            st.experimental_rerun()
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
        
        # Statistiques descriptives
        st.markdown("<h2>Statistiques Descriptives</h2>", unsafe_allow_html=True)
        st.dataframe(df[['grand_total', 'qty_ordered']].describe())
        
        # Top Catégories
        st.markdown("<h2>Top Catégories</h2>", unsafe_allow_html=True)
        top_categories = df['category_name_1'].value_counts().head(10)
        fig = px.bar(top_categories, 
                    title="Top 10 Catégories", 
                    labels={'value': 'Nombre de commandes', 'index': 'Catégorie'},
                    color_discrete_sequence=['#4e79a7'])
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribution des ventes
        st.markdown("<h2>Distribution des Ventes</h2>", unsafe_allow_html=True)
        fig = px.histogram(df, x='grand_total', nbins=50, title="Distribution des Montants des Commandes")
        st.plotly_chart(fig, use_container_width=True)

        if st.button("Calculer les Métriques RFM"):
            with st.spinner("Calcul RFM en cours..."):
                rfm_df = calculate_rfm(df)
                if rfm_df is not None:
                    rfm_df = create_rfm_segments(rfm_df)
                    st.session_state.rfm_df = rfm_df
                    st.success("✅ Analyse RFM terminée !")
                    st.experimental_rerun()

# Page Analyse RFM
elif selected_page == "Analyse RFM":
    if 'rfm_df' not in st.session_state:
        st.warning("⚠️ Veuillez d'abord calculer les métriques RFM.")
        if st.button("Calculer RFM"):
            if 'df' in st.session_state:
                with st.spinner("Calcul RFM en cours..."):
                    rfm_df = calculate_rfm(st.session_state.df)
                    if rfm_df is not None:
                        rfm_df = create_rfm_segments(rfm_df)
                        st.session_state.rfm_df = rfm_df
                        st.experimental_rerun()
            else:
                st.warning("Veuillez d'abord importer des données.")
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
        
        # Répartition des segments
        st.markdown("<h2>Répartition des Segments</h2>", unsafe_allow_html=True)
        segment_counts = rfm_df['Segment'].value_counts()
        fig = px.pie(segment_counts, 
                    values=segment_counts.values, 
                    names=segment_counts.index, 
                    title="Segments Clients")
        st.plotly_chart(fig, use_container_width=True)
        
        # Scatter plot RFM
        st.markdown("<h2>Relation RFM</h2>", unsafe_allow_html=True)
        fig = px.scatter(rfm_df, x='Recency', y='Frequency', size='Monetary', 
                        color='Segment', title="Relation Recency vs Frequency")
        st.plotly_chart(fig, use_container_width=True)

        if st.button("Effectuer le Clustering DBSCAN"):
            with st.spinner("Clustering en cours..."):
                rfm_clustered, silhouette_avg = perform_dbscan_clustering(rfm_df)
                st.session_state.rfm_clustered = rfm_clustered
                st.session_state.silhouette_avg = silhouette_avg
                st.success("✅ Clustering terminé !")
                st.experimental_rerun()

# Page Clustering
elif selected_page == "Clustering":
    if 'rfm_clustered' not in st.session_state:
        st.warning("⚠️ Veuillez d'abord effectuer le clustering RFM.")
        if st.button("Effectuer Clustering"):
            if 'rfm_df' in st.session_state:
                with st.spinner("Clustering en cours..."):
                    rfm_clustered, silhouette_avg = perform_dbscan_clustering(st.session_state.rfm_df)
                    st.session_state.rfm_clustered = rfm_clustered
                    st.session_state.silhouette_avg = silhouette_avg
                    st.experimental_rerun()
            else:
                st.warning("Veuillez d'abord calculer les métriques RFM.")
    else:
        rfm_clustered = st.session_state.rfm_clustered
        silhouette_avg = st.session_state.silhouette_avg
        st.markdown("<h1>Clustering DBSCAN</h1>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"<div class='metric-card'><h4>Score de Silhouette</h4><p>{silhouette_avg:.3f}</p></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='metric-card'><h4>Nombre de Clusters</h4><p>{rfm_clustered['Cluster'].nunique() - 1}</p></div>", unsafe_allow_html=True)

        st.markdown("---")
        
        # Visualisation 3D des clusters
        st.markdown("<h2>Visualisation 3D des Clusters</h2>", unsafe_allow_html=True)
        fig = px.scatter_3d(
            rfm_clustered,
            x='Recency',
            y='Frequency',
            z='Monetary',
            color='Cluster',
            title="Clusters DBSCAN",
            opacity=0.7
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribution par cluster
        st.markdown("<h2>Distribution par Cluster</h2>", unsafe_allow_html=True)
        cluster_stats = rfm_clustered.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
        st.dataframe(cluster_stats)

# Page Market Basket Analysis
elif selected_page == "Market Basket Analysis":
    if 'df' not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ Veuillez d'abord importer vos données.")
        if st.button("Aller à l'import"):
            st.session_state.page = "Import & Nettoyage"
            st.experimental_rerun()
    else:
        df = st.session_state.df
        st.markdown("<h1>Market Basket Analysis</h1>", unsafe_allow_html=True)

        min_support = st.slider("Support Minimum", 0.001, 0.1, 0.01, 0.001)
        
        if st.button("Analyser les Associations"):
            with st.spinner("Analyse en cours..."):
                frequent_itemsets, rules = perform_market_basket_analysis(df, min_support)

                st.markdown("<h2>Règles d'Association</h2>", unsafe_allow_html=True)
                if len(rules) > 0:
                    # Trier par lift décroissant
                    rules = rules.sort_values('lift', ascending=False)
                    st.dataframe(rules.head(10))
                    
                    # Top règles
                    st.markdown("<h2>Meilleures Associations</h2>", unsafe_allow_html=True)
                    top_rules = rules.head(5)
                    for i, rule in top_rules.iterrows():
                        antecedents = ", ".join(list(rule['antecedents']))
                        consequents = ", ".join(list(rule['consequents']))
                        st.markdown(f"**{antecedents}** → **{consequents}** (Confiance: {rule['confidence']:.2f}, Lift: {rule['lift']:.2f})")
                else:
                    st.warning("Aucune règle d'association trouvée avec ce support minimum.")

                st.markdown("---")
                st.markdown("<h2>Heatmap des Co-occurrences</h2>", unsafe_allow_html=True)
                basket = df.groupby(['increment_id', 'category_name_1'])['qty_ordered'].sum().unstack().fillna(0)
                cooccurrence_matrix = basket.T.dot(basket)
                fig = px.imshow(cooccurrence_matrix, 
                               labels=dict(x="Catégories", y="Catégories", color="Fréquence"),
                               title="Matrice de Co-occurrences")
                st.plotly_chart(fig, use_container_width=True)