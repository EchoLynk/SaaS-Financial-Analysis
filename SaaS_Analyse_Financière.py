#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import streamlit as st
import numpy as np
import numpy_financial as npf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration de la page
st.set_page_config(
    page_title="Analyse Financière SaaS",
    page_icon="📊",
    layout="wide"
)

# Style CSS personnalisé
st.markdown("""
    <style>
        .main > div {
            padding: 2rem 3rem;
        }
        .stProgress > div > div > div {
            background-color: #1f77b4;
        }
        .css-1v0mbdj {
            margin-top: 1.5rem;
        }
    </style>
""", unsafe_allow_html=True)

def calculate_financials(growth_rate, avg_price_per_user, opex_change, capex_value):
    # Données initiales
    initial_clients = 50
    initial_users_per_client = 10
    opex_base = 12400000
    variable_cost_per_user = 50
    amortization = capex_value / 5  # Amortissement sur 5 ans
    discount_rate = 0.10
    
    # Calcul du prêt NESDA (50% des CAPEX)
    nesda_loan = capex_value * 0.5
    annual_loan_payment = nesda_loan / 4  # Remboursement sur 4 ans à partir de l'année 2
    
    # Ajustement des OPEX
    opex = opex_base * (1 + opex_change / 100)
    
    # Initialisation des tableaux
    years = 5
    clients = [initial_clients]
    users = [initial_clients * initial_users_per_client]
    revenue = []
    variable_costs = []
    net_profit = []
    cashflow = []
    loan_payments = [0]  # Pas de remboursement en année 1
    
    # Remboursements annuels du prêt (années 2 à 5)
    for i in range(4):
        loan_payments.append(annual_loan_payment)
    
    # Calcul des valeurs annuelles
    for year in range(1, years + 1):
        if year > 1:
            clients.append(int(clients[-1] * (1 + growth_rate / 100)))
            users.append(int(clients[-1] * initial_users_per_client * (1 + growth_rate / 100)))
        
        revenue.append(users[year - 1] * avg_price_per_user * 12)
        variable_costs.append(users[year - 1] * variable_cost_per_user)
        net_profit.append(revenue[year - 1] - opex - variable_costs[year - 1] - amortization)
        cashflow.append(net_profit[year - 1] + amortization - loan_payments[year - 1])

    # Calculs financiers
    initial_investment = capex_value * 0.5
    npv = sum([cf / ((1 + discount_rate) ** (i + 1)) for i, cf in enumerate(cashflow)]) - initial_investment
    ip = (npv + initial_investment) / initial_investment
    irr = npf.irr([-initial_investment] + cashflow)
    total_cashflow = sum(cashflow)
    roi = (total_cashflow - initial_investment) / initial_investment * 100

    results = pd.DataFrame({
        "Année": range(1, years + 1),
        "Clients": clients,
        "Utilisateurs": users,
        "Chiffre d'affaires (DZD)": revenue,
        "Coûts variables (DZD)": variable_costs,
        "Remboursement prêt NESDA (DZD)": loan_payments,
        "Bénéfice net (DZD)": net_profit,
        "Cashflow (DZD)": cashflow
    })
    
    return results, npv, ip, irr, roi

def create_financial_charts(results):
    # Création des graphiques avec Plotly
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Évolution financière", "Croissance des clients et utilisateurs",
                       "Répartition des coûts", "Cashflow cumulé"),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )
    
    # Graphique 1: Évolution financière
    fig.add_trace(
        go.Scatter(x=results["Année"], y=results["Chiffre d'affaires (DZD)"],
                  name="Chiffre d'affaires", line=dict(color="#1f77b4")),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=results["Année"], y=results["Bénéfice net (DZD)"],
                  name="Bénéfice net", line=dict(color="#2ca02c")),
        row=1, col=1
    )
    
    # Graphique 2: Croissance
    fig.add_trace(
        go.Scatter(x=results["Année"], y=results["Clients"],
                  name="Clients", line=dict(color="#ff7f0e")),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=results["Année"], y=results["Utilisateurs"],
                  name="Utilisateurs", line=dict(color="#9467bd")),
        row=1, col=2
    )
    
    # Graphique 3: Répartition des coûts
    fig.add_trace(
        go.Bar(x=results["Année"], y=results["Coûts variables (DZD)"],
               name="Coûts variables", marker_color="#d62728"),
        row=2, col=1
    )
    fig.add_trace(
        go.Bar(x=results["Année"], y=results["Remboursement prêt NESDA (DZD)"],
               name="Remboursement NESDA", marker_color="#8c564b"),
        row=2, col=1
    )
    
    # Graphique 4: Cashflow cumulé
    cumulative_cashflow = np.cumsum(results["Cashflow (DZD)"])
    fig.add_trace(
        go.Scatter(x=results["Année"], y=cumulative_cashflow,
                  name="Cashflow cumulé", line=dict(color="#e377c2")),
        row=2, col=2
    )
    
    # Mise en page
    fig.update_layout(height=800, showlegend=True, template="plotly_white")
    
    return fig

# Sidebar pour les paramètres
with st.sidebar:
    st.image("ShuffleFlow.png", width=100)

    #st.image("C:/Users/guest/Desktop/analyse_MBA/ShuffleFlow.png", width=100)
    st.title("Paramètres")
    st.subheader("Paramètres de croissance")
    growth_rate = st.slider("Taux de croissance des clients (%)", 0, 100, 20)
    avg_price_per_user = st.number_input("Prix par utilisateur (DZD/mois)", 0, 10000, 1200)
    
    st.subheader("Paramètres financiers")
    opex_change = st.slider("Variation des OPEX (%)", -50, 100, 0)
    capex_value = st.number_input("CAPEX (DZD)", 1000000, 20000000, 6000000, step=100000)

# Corps principal
st.title("📊 Analyse Financière de Rentabilité SaaS")

# Calcul des résultats
results, npv, ip, irr, roi = calculate_financials(growth_rate, avg_price_per_user, opex_change, capex_value)

# Affichage des KPIs en colonnes
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("VAN", f"{npv:,.0f} DZD")
with col2:
    st.metric("IP", f"{ip:.2f}")
with col3:
    st.metric("TRI", f"{irr * 100:.1f}%")
with col4:
    st.metric("ROI", f"{roi:.1f}%")

# Structure de l'investissement
st.subheader("Structure de l'investissement")
invest_col1, invest_col2 = st.columns(2)
with invest_col1:
    st.info(f"💰 Apport personnel (50%): {capex_value*0.5:,.0f} DZD")
    st.info(f"🏦 Prêt NESDA (50%): {capex_value*0.5:,.0f} DZD")
with invest_col2:
    st.info(f"📊 CAPEX total: {capex_value:,.0f} DZD")
    st.info(f"💳 Remboursement annuel: {capex_value*0.5/4:,.0f} DZD")

# Graphiques
st.subheader("Analyses graphiques")
fig = create_financial_charts(results)
st.plotly_chart(fig, use_container_width=True)

# Téléchargement des résultats
st.subheader("Télécharger les résultats")

# Prepare the results for download
csv = results.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Télécharger les résultats financiers",
    data=csv,
    file_name='resultats_financiers.csv',
    mime='text/csv',
)

# Displaying the indicators for download
indicators = {
    "VAN": npv,
    "IP": ip,
    "TRI": irr * 100,
    "ROI": roi
}
indicators_df = pd.DataFrame(indicators, index=[0])

# Prepare the indicators for download
indicators_csv = indicators_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Télécharger les indicateurs financiers",
    data=indicators_csv,
    file_name='indicateurs_financiers.csv',
    mime='text/csv',
)

# Tableau détaillé
with st.expander("Voir le tableau détaillé des résultats"):
    st.dataframe(results.style.format({
        "Chiffre d'affaires (DZD)": "{:,.0f}",
        "Coûts variables (DZD)": "{:,.0f}",
        "Remboursement prêt NESDA (DZD)": "{:,.0f}",
        "Bénéfice net (DZD)": "{:,.0f}",
        "Cashflow (DZD)": "{:,.0f}"
    }))

# Notes explicatives
with st.expander("📝 Notes et explications"):
    st.markdown("""
    ### Indicateurs financiers
    - **VAN (Valeur Actuelle Nette)** : Mesure la rentabilité actualisée du projet
    - **IP (Indice de Profitabilité)** : Ratio de rentabilité par rapport à l'investissement
    - **TRI (Taux de Rendement Interne)** : Taux de rentabilité du projet
    - **ROI (Retour sur Investissement)** : Pourcentage de retour sur l'investissement initial
    
    ### Structure du financement
    - Financement mixte : 50% apport personnel, 50% prêt NESDA
    - Prêt NESDA à 0% d'intérêt
    - Remboursement sur 4 ans à partir de l'année 2
    - Indicateurs calculés sur la base de l'apport personnel
    """)

