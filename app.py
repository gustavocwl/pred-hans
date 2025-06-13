#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm
import warnings ; warnings.filterwarnings("ignore")

# ===============================
# Dados
df = pd.read_csv('dados_estimados.csv', sep=',')

# Criar indice
df = df.reset_index()

# Excluir observação final por estar com um número muito reduzido de casos
df = df[df['MES_DIAG'] != "2025-04"]

# Tratar formato
df['n_casos'] = pd.to_numeric(df['n_casos'], errors='coerce')

# Converter para datetime
df['MES_DIAG'] = pd.to_datetime(df['MES_DIAG'])

# Criar variável dummy para pandemia
df['pandemia'] = (df['MES_DIAG'] >= '2020-03-01') & (df['MES_DIAG'] <= '2023-05-01')

# Criar variável fase (pandemia em formato categórico)
df['fase'] = np.where(df['MES_DIAG'] < '2020-03-01', 1,
                      np.where((df['MES_DIAG'] >= '2020-03-01') & (df['MES_DIAG'] <= '2023-05-01'), 2,
                               np.where(df['MES_DIAG'] > '2023-05-01', 3,
                                        0)))
#
df['MES_DIAG'] = pd.to_datetime(df['MES_DIAG'], format='%Y-%m-%d')
df.set_index('MES_DIAG', inplace=True)
df = df.asfreq('MS')
df.index = pd.to_datetime(df.index)

# ===============================
# Treinar modelo SARIMAX com exógenas
endog = df["n_casos"]
exog = df[["fase", "n_casos_menor15", "n_casos_gif2", "n_casos_mb"]]

modelo = SARIMAX(endog, exog, order=(0, 1, 2), seasonal_order=(0, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False)
modelo_treinado = modelo.fit(disp=False)

# ===============================
# Streamlit
st.set_page_config(page_title="Hanseníase", layout="wide")
st.title("Calculadora para previsão de casos de hanseníase")

col1, col2 = st.columns([1, 5])

with col1:
    st.markdown("### Parâmetros")
    n_futuro = st.slider("Períodos futuros (meses)", 1, 36, 9)

    val_n_casos_menor15 = st.slider("Casos em menores de 15 anos", df['n_casos_menor15'].min().astype(int).item(), df['n_casos_menor15'].max().astype(int).item(), df['n_casos_menor15'].tail(12).mean().astype(int).item(), 1)
    val_n_casos_gif2 = st.slider("Casos diagnosticados com Grau 2 de Incapacidade Física (GIF2)", df['n_casos_gif2'].min().astype(int).item(), df['n_casos_gif2'].max().astype(int).item(), df['n_casos_gif2'].tail(12).mean().astype(int).item(), 1)
    val_n_casos_mb = st.slider("Casos multibacilar (MB)", df['n_casos_mb'].min().astype(int).item(), df['n_casos_mb'].max().astype(int).item(), df['n_casos_mb'].tail(12).mean().astype(int).item(), 1)

    mostrar_contrafactual = st.checkbox("Mostrar contrafatual", value=False)

with col2:
    # Criar DataFrame com valores futuros das variáveis
    datas_futuras = pd.date_range(df.index[-1] + pd.DateOffset(months=1), periods=n_futuro, freq="M")

    df_exog_futuro = pd.DataFrame({
        "fase": [3] * n_futuro,
        "n_casos_menor15": [val_n_casos_menor15] * n_futuro,
        "n_casos_gif2": [val_n_casos_gif2] * n_futuro,
        "n_casos_mb": [val_n_casos_mb] * n_futuro
    }, index=datas_futuras)

    # Previsão com intervalo de confiança
    forecast_obj = modelo_treinado.get_forecast(steps=n_futuro, exog=df_exog_futuro)
    previsoes = forecast_obj.predicted_mean
    ic = forecast_obj.conf_int(alpha=0.05)  # 95%

    df_prev = pd.DataFrame({
        "data": previsoes.index,
        "n_casos": previsoes.values,
        "lower": ic.iloc[:, 0],
        "upper": ic.iloc[:, 1],
        "tipo": "Previsto"
    }).round(0)

    df_obs = pd.DataFrame({
        "data": df.index,
        "n_casos": df["n_casos"].values,
        "n_casos_estimados": df["n_casos_estimados"].values,
        "n_casos_estimados_ic_l": df["n_casos_estimados_ic_l"].values,
        "n_casos_estimados_ic_u": df["n_casos_estimados_ic_u"].values,
        "tipo": "Observado"
    })

    # Gráfico
    fig = go.Figure()
    
    # Observados
    fig.add_trace(go.Scatter(
        x=df_obs["data"],
        y=df_obs["n_casos"],
        mode="lines+markers",
        name="Casos observados",
        #line=dict(color="darkblue")
    ))

    # Previstos
    fig.add_trace(go.Scatter(
        x=df_prev["data"],
        y=df_prev["n_casos"],
        mode="lines+markers",
        name="Casos previstos",
        line=dict(color="red")
    ))

    fig.add_trace(go.Scatter(
        x=df_prev["data"],
        y=df_prev["upper"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip"
    ))

    fig.add_trace(go.Scatter(
        x=df_prev["data"],
        y=df_prev["lower"],
        mode="lines",
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(255, 0, 0, 0.2)',
        name="Casos previstos (IC95%)"
    ))
    
    # Condicional para contrafatuais
    if mostrar_contrafactual:
        fig.add_trace(go.Scatter(
            x=df_obs["data"],
            y=df_obs["n_casos_estimados"],
            mode="lines+markers",
            name="Contrafatual",
            line=dict(color="white")
        ))

        fig.add_trace(go.Scatter(
            x=df_obs["data"],
            y=df_obs["n_casos_estimados_ic_l"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip"
        ))

        fig.add_trace(go.Scatter(
            x=df_obs["data"],
            y=df_obs["n_casos_estimados_ic_u"],
            mode="lines",
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(255, 255, 255, 0.2)',
            name="Contrafatual (IC95%)"
        ))

    fig.update_layout(
        #title="Casos de hanseníase: observados e previstos",
        xaxis_title="Data",
        yaxis_title="Casos",
        height=500,
        template="plotly_white",
        legend=dict(orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
    ))

    st.plotly_chart(fig, use_container_width=True)

st.write("")

col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1])

with col2: # Métrica
    st.metric(label=f"Casos (próx. {n_futuro} meses)", value=df_prev['n_casos'].astype(int).sum().item(), border=True)

with col3: # Métrica
    st.metric(label=f"Mínimo de casos (próx. {n_futuro} meses)", value=df_prev['lower'].astype(int).sum().item(), border=True)

with col4: # Métrica
    st.metric(label=f"Máximo de casos (próx. {n_futuro} meses)", value=df_prev['upper'].astype(int).sum().item(), border=True)

with col5: # Métrica
    st.metric(label=f"Média mensal de casos (próx. {n_futuro} meses)", value=int(df_prev['n_casos'].astype(int).sum().item() / n_futuro), border=True)

with col6: # Métrica: Teste de Independência dos Erros (Ljung-Box)
    def testar_independencia_erros(modelo_treinado):
        residuos = modelo_treinado.resid
        ljungbox_result = sm.stats.acorr_ljungbox(residuos, lags=[30], return_df=True)

        if ljungbox_result.iloc[0][1].item() < 0.05:
            return "👎" # Resíduos autocorrelacionados.
        else:
            return "👍" # Resíduos não autocorrelacionados.

    resultado = testar_independencia_erros(modelo_treinado)
    st.metric(label=f"Qualidade do modelo", value=resultado, border=True)

st.write("")

col1, col2 = st.columns([1, 5])

with col2:
    # Tabela
    df_prev = df_prev.set_index('data')
    df_prev.index = df_prev.index.astype(str).str[:7]
    st.dataframe(df_prev.transpose())

# In[ ]:




