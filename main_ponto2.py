import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from pypfopt import expected_returns, risk_models, EfficientFrontier

# Defina o universo de ativos (pode incluir ações, criptomoedas, etc.)
assets_universe = {
    'AAPL': 'Apple', 'MSFT': 'Microsoft', 'TSLA': 'Tesla',
    'GOOG': 'Alphabet', 'AMZN': 'Amazon', 'SPY': 'S&P 500 ETF',
    'BTC-USD': 'Bitcoin', 'ETH-USD': 'Ethereum', 'NVDA': 'NVIDIA', 'META': 'Meta'
}

# Entrada do usuário para o número de ativos na carteira
num_assets = st.number_input("Quantos ativos você deseja na sua carteira?", min_value=1, max_value=10, value=3)

# Baixar os dados históricos para todos os ativos do universo
st.write("Baixando dados históricos dos ativos...")

# Baixar os dados históricos (ajustados) de todos os ativos do universo
data = yf.download(list(assets_universe.keys()), start="2020-01-01", end="2024-01-01")['Adj Close']

# Calcular os retornos esperados e a matriz de covariância
mean_returns = expected_returns.mean_historical_return(data)
cov_matrix = risk_models.sample_cov(data)

# Criar o objeto Efficient Frontier
ef = EfficientFrontier(mean_returns, cov_matrix)

# Maximizar o Índice de Sharpe (sem restrições no número de ativos)
weights_all_assets = ef.max_sharpe()

# Ordenar os ativos por peso
sorted_weights = pd.Series(weights_all_assets).sort_values(ascending=False)

# Selecionar os 'num_assets' mais relevantes
selected_assets = sorted_weights.head(num_assets)

# Criar uma carteira considerando o número de ativos selecionados
ef = EfficientFrontier(mean_returns, cov_matrix)

# Forçar a otimização considerando apenas os ativos selecionados
ef_clean = EfficientFrontier(mean_returns[selected_assets.index], cov_matrix.loc[selected_assets.index, selected_assets.index])
weights_optimal = ef_clean.max_sharpe()

# Exibir os ativos selecionados e seus pesos
st.write(f"Ativos selecionados para a carteira ({num_assets} ativos):")
st.write(selected_assets)

# Exibir os pesos otimizados para os ativos
weights_optimal_series = pd.Series(weights_optimal, index=selected_assets.index)
st.write("Pesos otimizados para cada ativo:")
st.write(weights_optimal_series)

# Calcular o desempenho esperado da carteira
performance = ef_clean.portfolio_performance()
st.write(f"Retorno esperado da carteira: {performance[0]:.2f}%")
st.write(f"Risco (Desvio Padrão): {performance[1]:.2f}%")
st.write(f"Índice de Sharpe: {performance[2]:.2f}")

# Exibir um gráfico da carteira ótima com os pesos
st.bar_chart(weights_optimal_series)
