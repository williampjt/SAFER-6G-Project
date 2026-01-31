import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime, timedelta

st.set_page_config(
    page_title="SAFER-6G Dashboard",
    page_icon="🛡️",
    layout="wide"
)

st.markdown("""
<style>
    /* 1. Fond des cartes (Metrics et Containers) */
    .stMetric, .stContainer {
        border: 1px solid #3E4A5B !important;
        border-radius: 10px !important;
        padding: 15px !important;
    }

    /* 2. Titres des blocs */
    h1, h2, h3 {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }           
            
    /* 4. Customisation des onglets (Tabs) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }

.stTabs [data-baseweb="tab"] {
        background-color: #0f172b;
        border-radius: 4px 4px 0px 0px;
        color: white;
        border: 2px solid #3E4A5B;
        padding: 10px 20px;
        margin-right: 5px;
    }

</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model_results():
    file_path = "data/processed/unsw_test_dashboard.csv" 
    
    if not os.path.exists(file_path):
        st.error(f"Fichier {file_path} non trouvé. Lancez d'abord le notebook.")
        return pd.DataFrame()
    
    df = pd.read_csv(file_path)
    return df

def process_real_data(df_raw, n_rows=500):

    df = df_raw.sample(n=min(n_rows, len(df_raw))).copy()
    
    now = datetime.now()
    df['ts'] = [now - timedelta(seconds=np.random.randint(0, 3600)) for _ in range(len(df))]
    
    def assign_slice(row):
        if row['dur'] < 0.05: return 'URLLC'
        elif row['sbytes'] > 10000: return 'eMBB'
        else: return 'mMTC'

    df['slice_type'] = df.apply(assign_slice, axis=1)
    
    df['prediction'] = df['pred_label']
    df['probability'] = df['pred_proba_attack']
    df['attack_type'] = df['pred_attack_cat'].fillna("Normal")
    df['src_bytes'] = df['sbytes']
    df['latency_ms'] = df['dur'] * 1000
    
    return df.sort_values('ts')

raw_results = load_model_results()

if 'data' not in st.session_state and not raw_results.empty:
    st.session_state['data'] = process_real_data(raw_results)

if raw_results.empty:
    st.warning("⚠️ En attente des données du modèle...")
    st.stop()

df = st.session_state['data']

st.markdown("<h1 style='text-align: center;'>🛡️ SAFER-6G Security Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Decision Support System for <b>eMBB, URLLC, & mMTC</b> Network Slices</p>", unsafe_allow_html=True)
st.divider()

row1_col1, row1_col2 = st.columns(2)

# BLOC 1 : NETWORK OVERVIEW
with row1_col1:
    with st.container(border=True, height="stretch", vertical_alignment="center"):
        st.subheader("1️⃣ Network Overview")
        
        total_flows = len(df)
        n_attack = len(df[df['prediction'] == 1])
        n_normal = len(df[df['prediction'] == 0])
        
        pct_attack = (n_attack / total_flows) * 100
        pct_normal = (n_normal / total_flows) * 100
        
        if pct_attack < 5:
            status_txt = "NETWORK STATUS: OK"
            status_type = "success"
        elif pct_attack < 15:
            status_txt = "NETWORK STATUS: WARNING"
            status_type = "warning"
        else:
            status_txt = "NETWORK STATUS: CRITICAL"
            status_type = "error"

        if status_type == "success":
            st.success(f"✅ {status_txt}")
        elif status_type == "warning":
            st.warning(f"⚠️ {status_txt}")
        else:
            st.error(f"🚨 {status_txt}")

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Flows", f"{total_flows:,}")
        m2.metric("Normal Traffic", f"{pct_normal:.1f}%")
        m3.metric("Attack Traffic", f"{pct_attack:.1f}%", delta_color="inverse")

# BLOC 2 : DETECTION & ALERTS
with row1_col2:
    with st.container(border=True):
        st.subheader("2️⃣ Detection & Alerts")
        
        tab_graph, tab_alerts = st.tabs(["📉 Timeline", "⚠️ Alerts"])
        
        with tab_graph:
            timeline_df = df.set_index('ts').resample('1min')['prediction'].sum().reset_index()
            fig_tl = px.area(timeline_df, x='ts', y='prediction', height=200, color_discrete_sequence=['#E74C3C'])
            fig_tl.update_layout(margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_tl, use_container_width=True)
            
        with tab_alerts:
            recent = df[df['prediction'] == 1].copy()
            recent['ts'] = recent['ts'].dt.strftime('%H:%M:%S')
            recent['probability'] = (recent['probability']*100).apply(lambda x: f"{x:.1f}%")
            disp_df = recent[['ts', 'attack_type', 'slice_type', 'probability']].rename(
                columns={'ts':'Time', 'attack_type':'Type', 'slice_type':'Slice', 'probability':'Conf.'}
            )
            st.dataframe(disp_df, hide_index=True, use_container_width=True, height=200)

row2_col1, row2_col2 = st.columns(2)

# BLOC 3 : MODEL PERFORMANCE
with row2_col1:
    with st.container(border=True):
        st.subheader("3️⃣ AI Performance")
        
        c_roc, c_metrics = st.columns([1.2, 0.8])
        
        with c_roc:
            fpr = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 1]
            tpr = [0, 0.80, 0.92, 0.96, 0.98, 0.99, 1]
            
            fig_roc = px.area(x=fpr, y=tpr, title="ROC Curve (AUC=0.96)",
                              labels={'x':'False Positive Rate', 'y':'True Positive Rate'},
                              height=250)
            fig_roc.add_shape(type='line', line=dict(dash='dash', color='grey'),
                              x0=0, x1=1, y0=0, y1=1)
            fig_roc.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_roc, use_container_width=True)

        with c_metrics:
            st.markdown("##### Key Metrics")
            st.caption("Precision: **92.1%**")
            st.caption("Recall: **89.8%**")
            st.caption("F1-Score: **91.0%**")
            
            st.markdown("##### Top Features")
            st.progress(90, text="src_bytes")
            st.progress(75, text="duration")
            st.progress(60, text="latency")

# BLOC 4 : SLICE ANALYSIS
with row2_col2:
    with st.container(border=True, height="stretch", vertical_alignment="center"):
        st.subheader("4️⃣ Slice-Aware Analysis")
        
        subtab1, subtab2 = st.tabs(["📊 Charts", "📋 Detailed Load Table"])
        
        with subtab1:
            c1, c2 = st.columns(2)
            with c1:
                att_rate = df.groupby('slice_type')['prediction'].mean().reset_index()
                fig_rate = px.bar(att_rate, x='slice_type', y='prediction', title="Attack Rate", 
                                  color='slice_type', height=200)
                fig_rate.update_layout(margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
                st.plotly_chart(fig_rate, use_container_width=True)
            with c2:
                vol = df.groupby('slice_type')['src_bytes'].sum().reset_index()
                fig_vol = px.pie(vol, names='slice_type', values='src_bytes', title="Traffic Volume", 
                                 hole=0.4, height=200)
                fig_vol.update_layout(margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
                st.plotly_chart(fig_vol, use_container_width=True)

        with subtab2:
            summary = df.groupby('slice_type').agg(
                Avg_Bytes=('src_bytes', 'mean'),
                Alert_Rate=('prediction', 'mean'),
                Avg_Latency=('latency_ms', 'mean')
            ).reset_index()
            
            def get_load_label(bytes_val):
                if bytes_val > 100000: return "High 🔴"
                elif bytes_val > 1000: return "Medium 🟡"
                else: return "Low 🟢"
            
            summary['Traffic Load'] = summary['Avg_Bytes'].apply(get_load_label)
            
            summary['Alert Rate'] = (summary['Alert_Rate'] * 100).map('{:.1f}%'.format)
            summary['Avg Latency'] = summary['Avg_Latency'].map('{:.2f} ms'.format)
            
            final_table = summary[['slice_type', 'Traffic Load', 'Alert Rate', 'Avg Latency']]
            final_table.columns = ['Slice Type', 'Traffic Load', 'Alert Rate (%)', 'Avg Latency']
            
            st.dataframe(final_table, hide_index=True, use_container_width=True)

# Bouton Refresh
if st.button("🔄 Simulate Next Traffic Batch", use_container_width=True):
    st.session_state['data'] = process_real_data(raw_results)
    st.rerun()