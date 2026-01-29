import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(
    page_title="SAFER-6G Dashboard",
    page_icon="🛡️",
    layout="wide"
)

st.markdown("""
<style>
    .stMetric {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #e9ecef;
    }
    h1, h2, h3 { color: #2E4053; }
</style>
""", unsafe_allow_html=True)

def generate_mock_data(n_rows=500):
    slices = np.random.choice(['eMBB', 'URLLC', 'mMTC'], n_rows, p=[0.4, 0.2, 0.4])
    data = []
    
    for s in slices:
        timestamp = datetime.now() - timedelta(minutes=np.random.randint(0, 60))
        
        if s == 'eMBB': 
            src_bytes = np.random.randint(50000, 1000000)
            latency = np.random.uniform(5, 20)
            is_attack = np.random.choice([0, 1], p=[0.90, 0.10])
        elif s == 'URLLC': 
            src_bytes = np.random.randint(100, 2000)
            latency = np.random.uniform(0.1, 2.0)
            is_attack = np.random.choice([0, 1], p=[0.95, 0.05])
        else:
            src_bytes = np.random.randint(20, 5000)
            latency = np.random.uniform(20, 100)
            is_attack = np.random.choice([0, 1], p=[0.85, 0.15])

        prob = np.random.uniform(0.75, 0.99) if is_attack else np.random.uniform(0.01, 0.30)
        attack_type = np.random.choice(['DDoS', 'Scan', 'Injection']) if is_attack else "Normal"
        
        data.append({
            'ts': timestamp,
            'src_bytes': src_bytes,
            'duration': np.random.uniform(0.1, 5.0),
            'slice_type': s,
            'latency_ms': latency,
            'prediction': is_attack,
            'probability': prob,
            'attack_type': attack_type
        })
    
    return pd.DataFrame(data).sort_values('ts')

if 'data' not in st.session_state:
    st.session_state['data'] = generate_mock_data(500)
df = st.session_state['data']

st.title("🛡️ SAFER-6G Security Dashboard")
st.markdown("Decision Support System for **eMBB, URLLC, & mMTC** Network Slices")
st.divider()

row1_col1, row1_col2 = st.columns(2)

# BLOC 1 : NETWORK OVERVIEW
with row1_col1:
    with st.container(border=True):
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
        
        tab_graph, tab_alerts = st.tabs(["📉 Timeline", "⚠️ Recent Alerts"])
        
        with tab_graph:
            timeline_df = df.set_index('ts').resample('1min')['prediction'].sum().reset_index()
            fig_tl = px.area(timeline_df, x='ts', y='prediction', height=200, color_discrete_sequence=['#E74C3C'])
            fig_tl.update_layout(margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_tl, use_container_width=True)
            
        with tab_alerts:
            recent = df[df['prediction'] == 1].tail(5).copy()
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
    with st.container(border=True):
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
if st.button("🔄 Generate New 6G Traffic", use_container_width=True):
    st.session_state['data'] = generate_mock_data(500)
    st.rerun()