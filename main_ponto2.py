import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from pypfopt import expected_returns, risk_models, EfficientFrontier

# Defina o universo de ativos (você pode adicionar mais ativos aqui)
assets_universe = {
    'AAPL': 'Apple', 'MSFT': 'Microsoft', 'TSLA': 'Tesla',
    'GOOG': 'Alphabet', 'AMZN': 'Amazon', 'SPY': 'S&P 500 ETF',
    'BTC-USD': 'Bitcoin', 'ETH-USD': 'Ethereum'
}

# Entrada do usuário para o número de ativos na carteira
num_assets = st.number_input("Quantos ativos você deseja na sua carteira?", min_value=1, max_value=10, value=3)

# Seleção dos ativos com base no número escolhido
selected_assets = st.multiselect("Escolha os ativos", list(assets_universe.keys()), default=list(assets_universe.keys())[:num_assets])

# Baixar os dados históricos para os ativos selecionados
if selected_assets:
    try:
        # Baixar dados históricos
        data = yf.download(selected_assets, start="2020-01-01", end="2024-01-01")['Adj Close']
        
        # Exibir os dados
        st.write("Dados históricos de preços de fechamento ajustados:")
        st.dataframe(data)
        
        # Calcular os retornos esperados e a matriz de covariância
        mean_returns = expected_returns.mean_historical_return(data)
        cov_matrix = risk_models.sample_cov(data)
        
        # Criar o objeto Efficient Frontier
        ef = EfficientFrontier(mean_returns, cov_matrix)
        
        # Maximizar o índice de Sharpe
        weights = ef.max_sharpe()
        
        # Exibir os pesos da carteira ótima
        st.write("Pesos da carteira ótima:")
        st.write(weights)
        
        # Calcular o desempenho esperado da carteira
        performance = ef.portfolio_performance()
        st.write(f"Retorno esperado: {performance[0]:.2f}%")
        st.write(f"Risco (Desvio Padrão): {performance[1]:.2f}%")
        st.write(f"Índice de Sharpe: {performance[2]:.2f}")
        
        # Exibir um gráfico da carteira ótima
        st.bar_chart(weights)
        
    except Exception as e:
        st.error(f"Ocorreu um erro ao baixar os dados: {str(e)}")
