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
import openai
import io
import requests
import gdown
import base64

warnings.filterwarnings('ignore')

# ========================= CONFIGURATION =========================
# Clé API OpenAI (à définir directement)
OPENAI_API_KEY = st.secrets["openai"]["api_key"] # Remplacez par votre vraie clé

st.set_page_config(
    page_title="DataInsight Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS avec icônes Font Awesome et design épuré
st.markdown("""
<style>
    /* Import Font Awesome et Google Fonts */
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css');
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --primary: #2563eb;
        --primary-dark: #1d4ed8;
        --secondary: #f59e0b;
        --success: #059669;
        --warning: #d97706;
        --error: #dc2626;
        --bg-primary: #0f172a;
        --bg-secondary: #1e293b;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --border: #334155;
    }
    
    .main .block-container {
        padding: 1.5rem;
        max-width: 1400px;
    }
    
    /* Typography */
    h1, h2, h3, h4 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        margin-bottom: 1rem !important;
    }
    
    /* Header minimaliste */
    .app-header {
        background: linear-gradient(135deg, var(--primary), var(--primary-dark));
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 10px 25px rgba(37, 99, 235, 0.3);
    }
    
    .app-header h1 {
        color: white !important;
        margin: 0 !important;
        font-size: 2rem !important;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1rem;
    }
    
    /* Cards avec icônes */
    .metric-card {
        background: var(--bg-secondary);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid var(--border);
        text-align: center;
        transition: transform 0.2s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, var(--primary), var(--secondary));
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--primary);
        margin: 0.5rem 0;
        display: block;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 1px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
    }
    
    /* Navigation sidebar */
    .nav-item {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.75rem;
        margin: 0.25rem 0;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.2s ease;
        background: var(--bg-secondary);
        border: 1px solid var(--border);
    }
    
    .nav-item:hover {
        background: var(--primary);
        color: white;
    }
    
    .nav-icon {
        width: 20px;
        text-align: center;
    }
    
    /* Status badges */
    .status-badge {
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .status-success {
        background: var(--success);
        color: white;
    }
    
    .status-warning {
        background: var(--warning);
        color: white;
    }
    
    .status-info {
        background: var(--primary);
        color: white;
    }
    
    /* AI Section */
    .ai-section {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        color: white;
    }
    
    .ai-content {
        background: rgba(255,255,255,0.1);
        padding: 1rem;
        border-radius: 8px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
        margin-top: 1rem;
    }
    
    /* Upload area */
    .upload-zone {
        border: 2px dashed var(--primary);
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: rgba(37, 99, 235, 0.05);
        margin: 1rem 0;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary), var(--primary-dark));
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 500;
        transition: all 0.2s ease;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.4);
    }
    
    /* Hide default elements */
    .css-1rs6os, .css-17ziqus {
        visibility: hidden;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: var(--bg-secondary);
    }
</style>
""", unsafe_allow_html=True)

# ========================= ASSISTANT IA =========================
class AIAnalysisAssistant:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"
    
    def analyze_rfm_data(self, rfm_df, segment_stats):
        context = f"""
        RFM ANALYSIS DATA:
        Total clients: {len(rfm_df):,}
        Avg Recency: {rfm_df['Recency'].mean():.1f} days
        Avg Frequency: {rfm_df['Frequency'].mean():.2f}
        Avg Monetary: ${rfm_df['Monetary'].mean():.2f}
        
        Segments: {rfm_df['Segment'].value_counts().to_dict()}
        """
        
        prompt = """Provide concise RFM analysis focusing on:
        1. Key insights (2-3 points max)
        2. Priority actions (3 specific recommendations)
        3. Revenue opportunities
        
        Keep response under 300 words. Use bullet points."""
        
        return self._get_ai_response(prompt, context)
    
    def analyze_clustering_data(self, rfm_clustered, silhouette_score, n_clusters):
        cluster_stats = rfm_clustered.groupby('Cluster').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': ['mean', 'sum'],
            'customer_id': 'count'
        }).round(2)
        
        context = f"""
        CLUSTERING RESULTS:
        Clusters: {n_clusters}
        Silhouette Score: {silhouette_score:.3f}
        
        Cluster profiles: {cluster_stats.to_string()}
        """
        
        prompt = """Analyze clustering results. Provide:
        1. Quality assessment
        2. Cluster characteristics (1-2 sentences each)
        3. Strategic recommendations
        
        Keep under 250 words. Be actionable."""
        
        return self._get_ai_response(prompt, context)
    
    def analyze_market_basket(self, rules_df, frequent_items, baskets_count):
        top_rules = rules_df.head(5) if len(rules_df) > 0 else pd.DataFrame()
        
        context = f"""
        MARKET BASKET ANALYSIS:
        Baskets: {baskets_count:,}
        Rules found: {len(rules_df)}
        Top items: {dict(list(frequent_items.items())[:10]) if frequent_items else {}}
        
        Top rules: {top_rules[['Antecedent', 'Consequent', 'Lift']].to_string() if len(top_rules) > 0 else 'None'}
        """
        
        prompt = """Analyze market basket data. Focus on:
        1. Best opportunities for cross-selling
        2. Product bundling recommendations
        3. Revenue impact estimation
        
        Under 200 words. Prioritize actionable insights."""
        
        return self._get_ai_response(prompt, context)
    
    def _get_ai_response(self, prompt: str, context: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": context}
                ],
                max_tokens=800,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Analysis error: {str(e)}"

# ========================= DRIVE IMPORTER =========================
class DriveImporter:
    @staticmethod
    def extract_file_id(drive_url):
        patterns = [
            r'/file/d/([a-zA-Z0-9-_]+)',
            r'id=([a-zA-Z0-9-_]+)',
            r'/d/([a-zA-Z0-9-_]+)/view'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, drive_url)
            if match:
                return match.group(1)
        return None
    
    @staticmethod
    def download_from_drive(file_id):
        try:
            download_url = f'https://drive.google.com/uc?id={file_id}'
            output = io.BytesIO()
            gdown.download(download_url, output, quiet=True)
            output.seek(0)
            return output
        except Exception as e:
            st.error(f"Download error: {str(e)}")
            return None

# ========================= DATA FUNCTIONS =========================
@st.cache_data
def load_and_clean_data(data_source, source_type="file"):
    try:
        if source_type == "drive":
            df = pd.read_csv(data_source, low_memory=False)
        else:
            df = pd.read_csv(data_source, low_memory=False)
        
        # Clean columns
        cols_to_drop = ['Unnamed: 21', 'Unnamed: 22', 'Unnamed: 23', 'Unnamed: 24', 'Unnamed: 25']
        cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)
        
        # Rename columns
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
        
        # Payment method mapping
        payment_method_mapping = {
            'cod': 'direct', 'ublcreditcard': 'credit_card', 'mygateway': 'credit_card',
            'customercredit': 'direct', 'cashatdoorstep': 'direct', 'mcblite': 'mobile-money',
            'internetbanking': 'direct', 'marketingexpense': 'direct', 'productcredit': 'direct',
            'financesettlement': 'direct', 'Payaxis': 'credit_card', 'jazzvoucher': 'mobile-money',
            'jazzwallet': 'mobile-money', 'Easypay': 'mobile-money', 'Easypay_MA': 'mobile-money',
            'easypay_voucher': 'mobile-money', 'bankalfalah': 'direct', 'apg': 'credit_card',
            'Unknown': 'direct'
        }
        if 'payment_method' in df.columns:
            df['payment_method'] = df['payment_method'].map(payment_method_mapping).fillna('direct')
        
        # Clean essential data
        essential_cols = ['customer_id', 'created_at', 'grand_total', 'status']
        if 'category_name_1' in df.columns:
            essential_cols.append('category_name_1')
        
        df = df.dropna(subset=[col for col in essential_cols if col in df.columns])
        
        # Handle missing values
        if 'discount_amount' in df.columns:
            df['discount_amount'] = df['discount_amount'].fillna(0)
        if 'sales_commission_code' in df.columns:
            df['sales_commission_code'] = df['sales_commission_code'].fillna('Unknown')
        
        # Date conversion
        date_cols = ['created_at', 'customer_since', 'working_date']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Handle outliers
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
        st.error(f"Loading error: {e}")
        return None

@st.cache_data
def calculate_rfm(df):
    try:
        df_filtered = df.dropna(subset=['created_at'])
        latest_date = df_filtered['created_at'].max()
        
        recency_df = df_filtered.groupby('customer_id')['created_at'].max().reset_index()
        recency_df['Recency'] = (latest_date - recency_df['created_at']).dt.days
        
        rfm_df = df_filtered.groupby('customer_id').agg(
            Frequency=('increment_id', 'nunique'),
            Monetary=('grand_total', 'sum')
        ).reset_index()
        
        rfm_df = rfm_df.merge(recency_df[['customer_id', 'Recency']], on='customer_id')
        return rfm_df[['customer_id', 'Recency', 'Frequency', 'Monetary']]
        
    except Exception as e:
        st.error(f"RFM calculation error: {e}")
        return None

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
    rfm_vars = ['Recency', 'Frequency', 'Monetary']
    rfm_transformed = rfm_df.copy()
    
    for col in rfm_vars:
        rfm_transformed[col] = np.log1p(rfm_transformed[col])
    
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_transformed[rfm_vars])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(rfm_scaled)
    
    rfm_df['Cluster'] = clusters
    silhouette_avg = silhouette_score(rfm_scaled, clusters)
    
    return rfm_df, rfm_scaled, silhouette_avg, scaler, kmeans

def parse_items(cell):
    if pd.isna(cell):
        return []
    parts = [x.strip().strip('"').strip("'") for x in str(cell).split(",")]
    return [re.sub(r"\s+", " ", p).strip() for p in parts if p and p.strip()]

@st.cache_data
def prepare_market_basket_data(df, sample_size=50000):
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
        grouped = df_sample.groupby("increment_id")["category_name_1"].apply(list)
        for tid, categories in grouped.items():
            for cat in categories:
                if pd.notna(cat):
                    transactions[tid].add(str(cat))
    
    baskets = [sorted(list(items)) for tid, items in transactions.items() if len(items) > 1]
    return baskets, transactions

def simple_apriori(baskets, min_support=0.01, min_confidence=0.5):
    item_counts = defaultdict(int)
    for basket in baskets:
        for item in basket:
            item_counts[item] += 1
    
    total_baskets = len(baskets)
    min_support_count = int(min_support * total_baskets)
    
    frequent_items = {item: count for item, count in item_counts.items() 
                     if count >= min_support_count}
    
    rules = []
    for basket in baskets:
        basket_items = [item for item in basket if item in frequent_items]
        for i, item_a in enumerate(basket_items):
            for item_b in basket_items[i+1:]:
                rules.append((item_a, item_b, frequent_items[item_a], frequent_items[item_b]))
    
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

# ========================= INTERFACE =========================

# Header
st.markdown("""
<div class="app-header">
    <h1>
        <i class="fas fa-chart-line"></i>
        DataInsight Pro
    </h1>
</div>
""", unsafe_allow_html=True)

# Initialize AI Assistant
if OPENAI_API_KEY and OPENAI_API_KEY != "sk-your-openai-api-key-here":
    st.session_state.ai_assistant = AIAnalysisAssistant(OPENAI_API_KEY)
    ai_enabled = True
else:
    ai_enabled = False

# Sidebar Navigation
st.sidebar.markdown("### Navigation")

page_options = {
    "Dashboard": {"icon": "fas fa-tachometer-alt", "page": "dashboard"},
    "Data Import": {"icon": "fas fa-upload", "page": "import"},
    "RFM Analysis": {"icon": "fas fa-users", "page": "rfm"},
    "Clustering": {"icon": "fas fa-sitemap", "page": "clustering"},
    "Market Basket": {"icon": "fas fa-shopping-cart", "page": "basket"},
    "AI Analysis": {"icon": "fas fa-robot", "page": "ai"}
}

selected_page = st.sidebar.radio(
    "Select Section",
    list(page_options.keys()),
    format_func=lambda x: f"{page_options[x]['icon']} {x}" if 'icon' in page_options[x] else x
)

page = page_options[selected_page]["page"]

# Status indicators
st.sidebar.markdown("---")
st.sidebar.markdown("### Status")

if 'df' in st.session_state:
    st.sidebar.markdown('<div class="status-badge status-success"><i class="fas fa-check"></i> Data Loaded</div>', unsafe_allow_html=True)
    st.sidebar.metric("Records", f"{len(st.session_state.df):,}")
else:
    st.sidebar.markdown('<div class="status-badge status-warning"><i class="fas fa-exclamation-triangle"></i> No Data</div>', unsafe_allow_html=True)

if ai_enabled:
    st.sidebar.markdown('<div class="status-badge status-info"><i class="fas fa-robot"></i> AI Ready</div>', unsafe_allow_html=True)
else:
    st.sidebar.markdown('<div class="status-badge status-warning"><i class="fas fa-robot"></i> AI Disabled</div>', unsafe_allow_html=True)

# ========================= PAGES =========================

if page == "dashboard":
    if 'df' in st.session_state:
        df = st.session_state.df
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_revenue = df['grand_total'].sum()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label"><i class="fas fa-dollar-sign"></i> Revenue</div>
                <div class="metric-value">${total_revenue:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            avg_order = df['grand_total'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label"><i class="fas fa-shopping-bag"></i> Avg Order</div>
                <div class="metric-value">${avg_order:.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            total_customers = df['customer_id'].nunique()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label"><i class="fas fa-users"></i> Customers</div>
                <div class="metric-value">{total_customers:,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            total_orders = df['increment_id'].nunique() if 'increment_id' in df.columns else len(df)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label"><i class="fas fa-receipt"></i> Orders</div>
                <div class="metric-value">{total_orders:,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Quick visualization
        if 'created_at' in df.columns:
            st.markdown("### Revenue Trend")
            df_daily = df.groupby(df['created_at'].dt.date)['grand_total'].sum().reset_index()
            df_daily.columns = ['Date', 'Revenue']
            
            fig = px.line(df_daily, x='Date', y='Revenue', title="Daily Revenue")
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Import data to see dashboard metrics")

elif page == "import":
    st.markdown("### Data Import")
    
    import_method = st.radio(
        "Source",
        ["Local File", "Google Drive", "URL"]
    )
    
    if import_method == "Local File":
        st.markdown("""
        <div class="upload-zone">
            <i class="fas fa-file-csv fa-3x" style="color: var(--primary); margin-bottom: 1rem;"></i>
            <h4>Upload CSV File</h4>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("", type=["csv"])
        
        if uploaded_file:
            with st.spinner("Processing..."):
                df = load_and_clean_data(uploaded_file)
                
            if df is not None:
                st.success(f"Loaded {len(df):,} records")
                st.session_state.df = df
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Rows", f"{len(df):,}")
                    st.metric("Columns", f"{len(df.columns)}")
                
                with col2:
                    st.metric("Customers", f"{df['customer_id'].nunique():,}")
                    st.metric("Size", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    elif import_method == "Google Drive":
        st.markdown("""
        <div class="upload-zone">
            <i class="fab fa-google-drive fa-3x" style="color: var(--secondary); margin-bottom: 1rem;"></i>
            <h4>Import from Google Drive</h4>
        </div>
        """, unsafe_allow_html=True)
        
        drive_url = st.text_input("Drive Share URL")
        
        if drive_url and st.button("Import"):
            file_id = DriveImporter.extract_file_id(drive_url)
            
            if file_id:
                with st.spinner("Downloading..."):
                    drive_data = DriveImporter.download_from_drive(file_id)
                
                if drive_data:
                    df = load_and_clean_data(drive_data, "drive")
                    
                    if df is not None:
                        st.success(f"Imported {len(df):,} records from Drive")
                        st.session_state.df = df
            else:
                st.error("Invalid Drive URL")

elif page == "rfm":
    if 'df' not in st.session_state:
        st.warning("Import data first")
    else:
        st.markdown("### RFM Analysis")
        
        df = st.session_state.df
        
        with st.spinner("Calculating RFM..."):
            rfm_df = calculate_rfm(df)
        
        if rfm_df is not None:
            rfm_segments = create_rfm_segments(rfm_df)
            st.session_state.rfm_df = rfm_segments
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label"><i class="fas fa-clock"></i> Avg Recency</div>
                    <div class="metric-value">{rfm_df['Recency'].mean():.0f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label"><i class="fas fa-sync-alt"></i> Avg Frequency</div>
                    <div class="metric-value">{rfm_df['Frequency'].mean():.1f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label"><i class="fas fa-coins"></i> Avg Monetary</div>
                    <div class="metric-value">${rfm_df['Monetary'].mean():.0f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label"><i class="fas fa-layer-group"></i> Segments</div>
                    <div class="metric-value">{rfm_segments['Segment'].nunique()}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                segment_counts = rfm_segments['Segment'].value_counts()
                fig = px.pie(values=segment_counts.values, names=segment_counts.index,
                           title="Segment Distribution")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                segment_value = rfm_segments.groupby('Segment')['Monetary'].sum().sort_values()
                fig = px.bar(x=segment_value.values, y=segment_value.index,
                           orientation='h', title="Value by Segment")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Segment stats
            segment_stats = rfm_segments.groupby('Segment').agg({
                'Recency': 'mean',
                'Frequency': 'mean',
                'Monetary': ['mean', 'sum'],
                'customer_id': 'count'
            }).round(2)
            
            segment_stats.columns = ['Recency', 'Frequency', 'Avg Value', 'Total Value', 'Count']
            st.dataframe(segment_stats, use_container_width=True)

elif page == "clustering":
    if 'rfm_df' not in st.session_state:
        st.warning("Complete RFM analysis first")
    else:
        st.markdown("### Customer Clustering")
        
        rfm_df = st.session_state.rfm_df
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            n_clusters = st.slider("Clusters", 2, 8, 4)
            
            if st.button("Run Clustering"):
                with st.spinner("Clustering..."):
                    rfm_clustered, _, silhouette_avg, _, _ = perform_clustering(rfm_df, n_clusters)
                    st.session_state.rfm_clustered = rfm_clustered
                    st.session_state.silhouette_score = silhouette_avg
        
        if 'rfm_clustered' in st.session_state:
            rfm_clustered = st.session_state.rfm_clustered
            silhouette_avg = st.session_state.silhouette_score
            
            with col1:
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label"><i class="fas fa-chart-bar"></i> Silhouette</div>
                        <div class="metric-value">{silhouette_avg:.3f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_b:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label"><i class="fas fa-cubes"></i> Clusters</div>
                        <div class="metric-value">{n_clusters}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_c:
                    quality = "High" if silhouette_avg > 0.5 else "Medium" if silhouette_avg > 0.3 else "Low"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label"><i class="fas fa-star"></i> Quality</div>
                        <div class="metric-value" style="font-size: 1.5rem;">{quality}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.scatter(rfm_clustered, x='Frequency', y='Monetary',
                               color='Cluster', size='Recency',
                               title="Clusters: Frequency vs Monetary")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                cluster_counts = rfm_clustered['Cluster'].value_counts().sort_index()
                fig = px.bar(x=[f"C{i}" for i in cluster_counts.index], y=cluster_counts.values,
                           title="Cluster Distribution")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

elif page == "basket":
    if 'df' not in st.session_state:
        st.warning("Import data first")
    else:
        st.markdown("### Market Basket Analysis")
        
        df = st.session_state.df
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_support = st.slider("Min Support", 0.001, 0.1, 0.01, 0.001)
        with col2:
            min_confidence = st.slider("Min Confidence", 0.1, 1.0, 0.5, 0.1)
        with col3:
            sample_size = st.slider("Sample Size", 1000, 100000, 50000, 1000)
        
        if st.button("Analyze Associations"):
            with st.spinner("Processing baskets..."):
                baskets, transactions = prepare_market_basket_data(df, sample_size)
                
            if baskets:
                st.session_state.baskets = baskets
                
                with st.spinner("Finding rules..."):
                    rules, frequent_items = simple_apriori(baskets, min_support, min_confidence)
                    st.session_state.rules_df = pd.DataFrame(rules) if rules else pd.DataFrame()
                    st.session_state.frequent_items = frequent_items
                
                st.success(f"Analyzed {len(baskets):,} baskets, found {len(rules)} rules")
        
        if 'rules_df' in st.session_state and len(st.session_state.rules_df) > 0:
            rules_df = st.session_state.rules_df.sort_values('Lift', ascending=False)
            frequent_items = st.session_state.frequent_items
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label"><i class="fas fa-list"></i> Rules</div>
                    <div class="metric-value">{len(rules_df)}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label"><i class="fas fa-boxes"></i> Items</div>
                    <div class="metric-value">{len(frequent_items)}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label"><i class="fas fa-arrow-up"></i> Max Lift</div>
                    <div class="metric-value">{rules_df['Lift'].max():.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label"><i class="fas fa-percentage"></i> Avg Confidence</div>
                    <div class="metric-value">{rules_df['Confidence'].mean():.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Top rules
            st.markdown("#### Top Association Rules")
            display_rules = rules_df.head(10)[['Antecedent', 'Consequent', 'Support', 'Confidence', 'Lift']]
            st.dataframe(display_rules, use_container_width=True)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.scatter(rules_df, x='Support', y='Confidence', size='Lift',
                               title="Support vs Confidence")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                frequent_items_df = pd.DataFrame(
                    list(frequent_items.items()), 
                    columns=['Item', 'Frequency']
                ).sort_values('Frequency', ascending=False).head(10)
                
                fig = px.bar(frequent_items_df, x='Frequency', y='Item',
                           orientation='h', title="Top Items")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

elif page == "ai":
    if not ai_enabled:
        st.warning("Configure OpenAI API key in code")
    else:
        st.markdown("""
        <div class="ai-section">
            <h3><i class="fas fa-robot"></i> AI Analysis</h3>
            <div class="ai-content">
                Select an analysis type to get AI-powered insights and recommendations.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        analysis_type = st.selectbox(
            "Analysis Type",
            ["RFM Analysis", "Clustering", "Market Basket"]
        )
        
        if analysis_type == "RFM Analysis":
            if 'rfm_df' not in st.session_state:
                st.warning("Complete RFM analysis first")
            else:
                rfm_df = st.session_state.rfm_df
                segment_stats = rfm_df.groupby('Segment').agg({
                    'Recency': 'mean',
                    'Frequency': 'mean',
                    'Monetary': ['mean', 'sum'],
                    'customer_id': 'count'
                }).round(2)
                
                if st.button("Generate AI Analysis"):
                    with st.spinner("AI analyzing RFM data..."):
                        ai_analysis = st.session_state.ai_assistant.analyze_rfm_data(rfm_df, segment_stats)
                    
                    st.markdown("#### AI Insights")
                    st.markdown(ai_analysis)
        
        elif analysis_type == "Clustering":
            if 'rfm_clustered' not in st.session_state:
                st.warning("Complete clustering first")
            else:
                rfm_clustered = st.session_state.rfm_clustered
                silhouette_score = st.session_state.silhouette_score
                n_clusters = rfm_clustered['Cluster'].nunique()
                
                if st.button("Generate AI Analysis"):
                    with st.spinner("AI analyzing clusters..."):
                        ai_analysis = st.session_state.ai_assistant.analyze_clustering_data(
                            rfm_clustered, silhouette_score, n_clusters)
                    
                    st.markdown("#### AI Insights")
                    st.markdown(ai_analysis)
        
        elif analysis_type == "Market Basket":
            if 'rules_df' not in st.session_state or len(st.session_state.rules_df) == 0:
                st.warning("Complete market basket analysis first")
            else:
                rules_df = st.session_state.rules_df
                frequent_items = st.session_state.frequent_items
                baskets_count = len(st.session_state.baskets)
                
                if st.button("Generate AI Analysis"):
                    with st.spinner("AI analyzing associations..."):
                        ai_analysis = st.session_state.ai_assistant.analyze_market_basket(
                            rules_df, frequent_items, baskets_count)
                    
                    st.markdown("#### AI Insights")
                    st.markdown(ai_analysis)
        
        # Custom questions
        st.markdown("---")
        st.markdown("#### Custom Question")
        
        custom_question = st.text_area("Ask a specific question about your data")
        
        if custom_question and st.button("Get Answer"):
            context_parts = []
            
            if 'df' in st.session_state:
                df = st.session_state.df
                context_parts.append(f"""
                Data Overview:
                - Customers: {df['customer_id'].nunique():,}
                - Revenue: ${df['grand_total'].sum():,.2f}
                - Orders: {df['increment_id'].nunique():,}
                """)
            
            if 'rfm_df' in st.session_state:
                context_parts.append(f"RFM segments: {st.session_state.rfm_df['Segment'].value_counts().to_dict()}")
            
            full_context = "\n".join(context_parts)
            
            if full_context:
                try:
                    with st.spinner("Generating response..."):
                        response = st.session_state.ai_assistant._get_ai_response(
                            "Answer this e-commerce question concisely with actionable advice.",
                            f"{full_context}\n\nQuestion: {custom_question}"
                        )
                    
                    st.markdown("#### AI Response")
                    st.markdown(response)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            else:
                st.warning("No analyzed data available")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; color: var(--text-secondary);">
    <i class="fas fa-chart-line"></i> DataInsight Pro - E-commerce Analytics Platform
</div>
""", unsafe_allow_html=True)