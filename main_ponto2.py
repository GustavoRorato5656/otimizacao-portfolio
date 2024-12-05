import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns
import streamlit as st
import pandas as pd
import numpy as np

# Definir o universo de ativos
UNIVERSO_ATIVOS = {
    "Criptomoedas": ["BTC-USD", "ETH-USD"],
    "Renda Fixa Brasil": ["IRFM11.SA", "IMAB11.SA"],
    "Ações Brasil": ["PETR4.SA", "VALE3.SA"],
    "Ações EUA": ["AAPL", "MSFT"],
    "Ações China": ["BABA", "JD"]
}

# Função para obter dados históricos
def get_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)
    return data['Adj Close']

# Função para otimização de portfólio
def optimize_portfolio(data, num_assets):
    mu = expected_returns.mean_historical_return(data)
    S = risk_models.sample_cov(data)
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    
    # Selecionar os ativos com os maiores pesos
    sorted_weights = sorted(cleaned_weights.items(), key=lambda x: x[1], reverse=True)
    top_assets = sorted_weights[:num_assets]
    top_weights = {asset: weight for asset, weight in top_assets}
    
    # Calcular métricas de desempenho
    performance = ef.portfolio_performance(verbose=True)
    
    return top_weights, performance

# Função para criar o dashboard
def create_dashboard():
    st.title("Sugestão de Carteira Ótima")
    
    # Seleção do número de ativos
    num_assets = st.number_input("Número de ativos na carteira", min_value=1, max_value=len(UNIVERSO_ATIVOS), value=3)
    
    # Seleção do horizonte de tempo
    start_date = st.date_input("Data de início")
    end_date = st.date_input("Data de término")
    
    if st.button("Calcular"):
        # Obter dados históricos para todos os ativos no universo
        tickers = [ticker for sublist in UNIVERSO_ATIVOS.values() for ticker in sublist]
        data = get_data(tickers, start_date, end_date)
        
        # Filtrar dados para os ativos selecionados
        selected_data = data[[ticker for ticker, _ in sorted_weights[:num_assets]]]
        
        # Otimizar portfólio
        top_weights, performance = optimize_portfolio(selected_data, num_assets)
        
        # Exibir resultados
        st.write("Pesos Ótimos:", top_weights)
        st.bar_chart(pd.DataFrame.from_dict(top_weights, orient='index', columns=['Peso']))
        
        # Exibir métricas de desempenho
        st.write(f"Retorno Esperado: {performance[0]:.2%}")
        st.write(f"Risco (Desvio Padrão): {performance[1]:.2%}")
        st.write(f"Índice de Sharpe: {performance[2]:.2f}")

if __name__ == "__main__":
    create_dashboard()
