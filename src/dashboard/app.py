import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score
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
    file_path = "model/data/processed/unsw_test_dashboard.csv" 
    
    if not os.path.exists(file_path):
        st.error(f"Fichier {file_path} non trouvé. Lancez d'abord le notebook.")
        return pd.DataFrame()
    
    df = pd.read_csv(file_path)
    return df

def process_real_data(df_raw, target_duration=3600):
    max_idx = max(0, len(df_raw) - 10000)
    start_idx = np.random.randint(0, max_idx)
    potential_data = df_raw.iloc[start_idx:].copy()
    potential_data['cum_dur'] = potential_data['dur'].cumsum()
    df = potential_data[potential_data['cum_dur'] <= target_duration].copy()
    
    if len(df) < 50: df = potential_data.head(50).copy()
    
    base_time = datetime.now() - timedelta(hours=1)
    df['ts'] = df['cum_dur'].apply(lambda x: base_time + timedelta(seconds=x))
    
    def assign_slice(row):
        if row['dur'] < 0.05: return 'URLLC'
        elif row['sbytes'] > 10000: return 'eMBB'
        else: return 'mMTC'
    df['slice_type'] = df.apply(assign_slice, axis=1)

    attack_map = {
        'normal': 'Normal',
        '0': 'DoS',
        '1': 'Exploits',
        '2': 'Fuzzers',
        '3': 'Generic',
        '4': 'Info_Gathering',
        '5': 'Malware'
    }

    df['prediction'] = df['pred_label']
    df['attack_type'] = df['pred_attack_cat'].map(attack_map).fillna("Normal")
    df['latency_ms'] = df['dur'] * 1000
    
    action_map = {
        'DoS': 'Rate Limiting & IP Blocking',
        'Exploits': 'System Patching & Virtual Patching',
        'Fuzzers': 'Sanitize Inputs & Protocol Filtering',
        'Generic': 'Deep Packet Inspection (DPI)',
        'Info_Gathering': 'Restrict Port Access & Obfuscation',
        'Malware': 'Isolate Host & Sandbox Execution',
    }
    
    def get_severity(row):
        if row['prediction'] == 0: return "Low"
        if row['attack_type'] in ['Fuzzers', 'Generic']: return "Medium"
        if row['attack_type'] in ['DoS', 'Exploits','Malware']: return "High"
        return "Low"

    df['recommended_action'] = df['attack_type'].map(action_map).fillna("Monitor")
    df['severity_level'] = df.apply(get_severity, axis=1)

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

        st.divider()
        
        st.markdown("**🛡️ Network Health Evolution (1h)**")
        health_df = df.set_index('ts').resample('1min')['prediction'].mean().reset_index()
        health_df['Health Score'] = (1 - health_df['prediction']) * 100
        
        fig_health = px.line(health_df, x='ts', y='Health Score', 
                             markers=True, height=200)
        
        fig_health.update_traces(line_color='#00D4FF', fill='tozeroy', fillcolor='rgba(0, 212, 255, 0.1)')
        
        fig_health.add_hrect(y0=95, y1=100, line_width=0, fillcolor="green", opacity=0.1, annotation_text="Safe")
        fig_health.add_hrect(y0=85, y1=95, line_width=0, fillcolor="orange", opacity=0.1, annotation_text="Warning")
        fig_health.add_hrect(y0=0, y1=85, line_width=0, fillcolor="red", opacity=0.1, annotation_text="Critical")
        
        fig_health.update_layout(template="plotly_dark", 
                                 paper_bgcolor='rgba(0,0,0,0)', 
                                 plot_bgcolor='rgba(0,0,0,0)',
                                 margin=dict(l=0, r=0, t=10, b=0),
                                 yaxis=dict(
                                    range=[0, 100], 
                                    title="Health Index"
                                ))
        
        st.plotly_chart(fig_health, use_container_width=True)

# BLOC 2 : DETECTION & ALERTS
with row1_col2:
    with st.container(border=True, height="stretch", vertical_alignment="center"):
        st.subheader("2️⃣ Detection & Alerts")
        
        tab_graph, tab_alerts = st.tabs(["📉 Timeline", "⚠️ Alerts"])
        
        with tab_graph:
            timeline_df = df.set_index('ts').resample('1min')['prediction'].sum().reset_index()
            fig_tl = px.area(timeline_df, x='ts', y='prediction', height=200, color_discrete_sequence=['#E74C3C'])
            fig_tl.update_layout(margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_tl, use_container_width=True)
            
with tab_alerts:
            alerts = df[df['prediction'] == 1].copy()
            alerts['Time'] = alerts['ts'].dt.strftime('%H:%M:%S')
            
            disp_df = alerts[['Time', 'attack_type', 'slice_type', 'severity_level', 'recommended_action', 'pred_proba_attack']]
            disp_df['pred_proba_attack'] = disp_df['pred_proba_attack'] * 100
            disp_df.columns = ['Time', 'Type', 'Slice', 'Severity', 'Recommended Action', 'Conf.']
            
            def style_severity(val):
                if val == "High":
                    return 'background-color: #721c24; color: #f8d7da; font-weight: bold;'
                elif val == "Medium":
                    return 'background-color: #856404; color: #fff3cd; font-weight: bold;'
                elif val == "Low":
                    return 'background-color: #155724; color: #d4edda; font-weight: bold;'

            styled_df = disp_df.style.applymap(style_severity, subset=['Severity'])

            st.dataframe(
                styled_df, 
                hide_index=True, 
                use_container_width=True, 
                height=250,
                column_config={
                    "Conf.": st.column_config.ProgressColumn(
                        "Confidence", min_value=0, max_value=100, format="%.1f%%"
                    )
                }
            )

row2_col1, row2_col2 = st.columns(2)

# BLOC 3 : MODEL PERFORMANCE
with row2_col1:
    with st.container(border=True, height="stretch", vertical_alignment="center"):
        st.subheader("3️⃣ AI Performance")
        
        c_roc, c_metrics = st.columns([1.2, 0.8])
        
        with c_roc:
            fpr, tpr, _ = roc_curve(raw_results['label'], raw_results['pred_proba_attack'])
            roc_auc = auc(fpr, tpr)
                
            fig_roc = px.area(
                x=fpr[::10], y=tpr[::10],
                title=f"Real-time ROC Curve (AUC={roc_auc:.3f})",
                labels={'x':'False Positive Rate', 'y':'True Positive Rate'},
                height=250)

            fig_roc.add_shape(
                type='line', line=dict(dash='dash', color='white'),
                x0=0, x1=1, y0=0, y1=1
            )

            fig_roc.update_layout(
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig_roc, use_container_width=True)
        
            st.markdown("##### Key Metrics")
            st.caption("Precision: **93%**")
            st.caption("Recall: **93%**")
            st.caption("F1-Score: **93%**")

    with c_metrics:
            st.markdown("##### Per-Slice Metrics")

            @st.cache_data
            def get_slice_metrics(df_raw_slice):
                y_true = df_raw_slice['label']
                y_pred = df_raw_slice['pred_label']
                
                p = precision_score(y_true, y_pred, zero_division=0)*100
                r = recall_score(y_true, y_pred, zero_division=0)*100
                f1 = f1_score(y_true, y_pred, zero_division=0)*100
                return p, r, f1

            slice_stats = []
            if 'slice_type' not in raw_results.columns:
                raw_results['slice_type'] = raw_results.apply(
                    lambda x: 'URLLC' if x['dur'] < 0.05 else ('eMBB' if x['sbytes'] > 10000 else 'mMTC'), axis=1
                )
                
            for s_type in ['URLLC', 'eMBB', 'mMTC']:
                global_subset = raw_results[raw_results['slice_type'] == s_type]
                if not global_subset.empty:
                    p, r, f1 = get_slice_metrics(global_subset)
                    slice_stats.append({"Slice": s_type, "Precision": p, "Recall": r, "F1": f1})
            
            perf_df = pd.DataFrame(slice_stats)

            st.dataframe(
                perf_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Precision": st.column_config.NumberColumn(format="%.1f%%"),
                    "Recall": st.column_config.NumberColumn(format="%.1f%%"),
                    "F1-Score": st.column_config.NumberColumn(format="%.1f%%"),
                }
            )

            st.markdown("##### Feature Importance")
            st.progress(57, text="num_ct_state_ttl")
            st.progress(20, text="num_sttl")
            st.progress(3, text="cat_service_dns")
            st.progress(2, text="cat_proto_arp")
            st.progress(2, text="num_dttl")

# BLOC 4 : SLICE ANALYSIS
with row2_col2:
    with st.container(border=True, height="stretch", vertical_alignment="center"):
        st.subheader("4️⃣ Slice-Aware Analysis")
        
        subtab1, subtab2, subtab3 = st.tabs(["📊 Charts", "📋 Detailed Load Table","🔥 Attack Heatmap"])
        
        with subtab1:
            c1, c2 = st.columns(2)
            with c1:
                att_rate = df.groupby('slice_type')['prediction'].mean().reset_index()
                fig_rate = px.bar(att_rate, x='slice_type', y='prediction', title="Attack Rate", 
                                  color='slice_type', height=200)
                fig_rate.update_layout(margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
                st.plotly_chart(fig_rate, use_container_width=True)
            with c2:
                vol = df.groupby('slice_type')['sbytes'].sum().reset_index()
                fig_vol = px.pie(vol, names='slice_type', values='sbytes', title="Traffic Volume", 
                                 hole=0.4, height=200)
                fig_vol.update_layout(margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
                st.plotly_chart(fig_vol, use_container_width=True)

        with subtab2:
            summary = df.groupby('slice_type').agg(
                Avg_Bytes=('sbytes', 'mean'),
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

        with subtab3:
            attacks_only = df[df['prediction'] == 1]
            
            if not attacks_only.empty:
                heat_data = attacks_only.groupby(['slice_type', 'attack_type']).size().reset_index(name='Count')
                
                custom_colors = [
                    [0.0, "#F8F8F8"],
                    [0.3, "#E7E70C"],
                    [0.6, "#FFA500"],
                    [1.0, "#FF0000"]
                ]
                
                fig_heat = px.density_heatmap(
                    heat_data, 
                    x='slice_type', 
                    y='attack_type', 
                    z='Count',
                    color_continuous_scale=custom_colors,
                    labels={'slice_type': 'Network Slice', 'attack_type': 'Attack Type'},
                    height=350
                )
                
                fig_heat.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=20, r=20, t=30, b=20),
                    coloraxis_colorbar=dict(title="Alerts")
                )
                
                st.plotly_chart(fig_heat, use_container_width=True)
            else:
                st.info("No attacks detected to display heatmap.")

# Bouton Refresh
if st.button("🔄 Simulate Next Traffic Batch", use_container_width=True):
    st.session_state['data'] = process_real_data(raw_results)
    st.rerun()