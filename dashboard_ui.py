# Nome do arquivo: dashboard_ui.py
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    PROMPTBOX DASHBOARD - B2B Analytics                        ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import plotly.express as px


def render_admin_view(conn):
    """Renderiza o Painel de Controle B2B."""
    
    # CSS Específico do Dashboard
    st.markdown("""
    <style>
        .kpi-card { 
            background-color: #1e293b; 
            padding: 20px; 
            border-radius: 10px; 
            border: 1px solid #334155; 
            text-align: center; 
        }
        .kpi-val { font-size: 24px; font-weight: bold; color: #fff; }
        .kpi-lbl { font-size: 12px; color: #94a3b8; text-transform: uppercase; }
        .status-ok { color: #10b981; }
        .status-warn { color: #f59e0b; }
    </style>
    """, unsafe_allow_html=True)

    st.title("📊 PromptBox Admin | B2B Cockpit")
    st.caption("Monitoramento de Soberania & Performance")

    # 1. Carregar Dados
    try:
        df = pd.read_sql_query("SELECT * FROM logs ORDER BY timestamp DESC", conn)
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        st.info("💡 Faça algumas perguntas no chat para popular o banco de dados.")
        return

    if df.empty:
        st.warning("Nenhum dado registrado ainda.")
        st.info("💡 Faça perguntas no modo Assistente Jurídico para começar a coletar métricas.")
        return

    # 2. KPIs
    total = len(df)
    latencia_media = df['response_time'].mean()
    latencia_p95 = df['response_time'].quantile(0.95) if len(df) > 1 else latencia_media
    
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.markdown(f'''
            <div class="kpi-card">
                <div class="kpi-val">{total}</div>
                <div class="kpi-lbl">Consultas Totais</div>
            </div>
        ''', unsafe_allow_html=True)
    
    with c2:
        color = "status-ok" if latencia_media < 30 else "status-warn"
        st.markdown(f'''
            <div class="kpi-card">
                <div class="kpi-val {color}">{latencia_media:.1f}s</div>
                <div class="kpi-lbl">Latência Média</div>
            </div>
        ''', unsafe_allow_html=True)
    
    with c3:
        st.markdown(f'''
            <div class="kpi-card">
                <div class="kpi-val">{latencia_p95:.1f}s</div>
                <div class="kpi-lbl">Latência P95</div>
            </div>
        ''', unsafe_allow_html=True)
    
    with c4:
        st.markdown('''
            <div class="kpi-card">
                <div class="kpi-val status-ok">100%</div>
                <div class="kpi-lbl">Dados Offline</div>
            </div>
        ''', unsafe_allow_html=True)

    st.markdown("---")

    # 3. Gráficos
    col_g1, col_g2 = st.columns([2, 1])
    
    with col_g1:
        st.subheader("📈 Volume de Uso por Hora")
        try:
            df['hora'] = pd.to_datetime(df['timestamp']).dt.hour
            hourly = df.groupby('hora').size().reset_index(name='qtd')
            
            fig = px.bar(
                hourly, 
                x='hora', 
                y='qtd',
                labels={'hora': 'Hora do Dia', 'qtd': 'Requisições'}
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", 
                plot_bgcolor="rgba(0,0,0,0)", 
                font_color="white",
                xaxis=dict(dtick=1)
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Erro ao gerar gráfico: {e}")

    with col_g2:
        st.subheader("🤖 Modelos Usados")
        if 'model_used' in df.columns and df['model_used'].notna().any():
            fig2 = px.pie(
                df, 
                names='model_used', 
                hole=0.4
            )
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", 
                font_color="white",
                showlegend=True
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Sem dados de modelo ainda.")

    # 4. Gráfico de Latência ao Longo do Tempo
    st.subheader("⏱️ Latência ao Longo do Tempo")
    try:
        df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
        fig3 = px.line(
            df.sort_values('timestamp_dt'), 
            x='timestamp_dt', 
            y='response_time',
            labels={'timestamp_dt': 'Timestamp', 'response_time': 'Tempo (s)'}
        )
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)", 
            font_color="white"
        )
        st.plotly_chart(fig3, use_container_width=True)
    except Exception as e:
        st.error(f"Erro: {e}")

    # 5. Logs de Auditoria
    st.subheader("🛡️ Logs de Auditoria")
    
    # Filtros
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        search = st.text_input("🔍 Buscar nas queries:", "")
    with col_f2:
        limit = st.selectbox("Exibir:", [10, 25, 50, 100], index=0)
    
    # Aplica filtros
    df_filtered = df.copy()
    if search:
        df_filtered = df_filtered[df_filtered['query'].str.contains(search, case=False, na=False)]
    
    # Exibe tabela
    st.dataframe(
        df_filtered[['timestamp', 'query', 'response_time', 'model_used']].head(limit),
        use_container_width=True, 
        hide_index=True,
        column_config={
            "timestamp": "⏰ Horário",
            "query": "💬 Pergunta",
            "response_time": st.column_config.NumberColumn("⏱️ Tempo (s)", format="%.1f"),
            "model_used": "🤖 Modelo"
        }
    )

    # 6. Exportar Dados
    st.markdown("---")
    st.subheader("📥 Exportar Dados")
    
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Baixar CSV",
            data=csv,
            file_name="promptbox_logs.csv",
            mime="text/csv"
        )
    with col_e2:
        st.info(f"📊 {len(df)} registros disponíveis")