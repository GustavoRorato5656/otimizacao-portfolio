import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from pypfopt import expected_returns, risk_models, EfficientFrontier
from random import sample

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

# Selecionar aleatoriamente os ativos para a carteira com o número de ativos desejado
selected_assets = sample(list(assets_universe.keys()), num_assets)

# Exibir os ativos selecionados
st.write(f"Ativos selecionados para a carteira ({num_assets} ativos):")
st.write([assets_universe[ticker] for ticker in selected_assets])

# Criar o objeto Efficient Frontier para otimização com os ativos selecionados
ef = EfficientFrontier(mean_returns[selected_assets], cov_matrix.loc[selected_assets, selected_assets])

# Maximizar o Índice de Sharpe (com os ativos selecionados)
weights_optimal = ef.max_sharpe()

# Verificar se algum ativo tem peso zero e forçar um peso mínimo
for ticker in weights_optimal:
    if weights_optimal[ticker] < 0.01:  # Se o peso for menor que 1%
        weights_optimal[ticker] = 0.01  # Atribui um peso mínimo para garantir a diversificação

# Normalizar os pesos para garantir que a soma seja 1
total_weight = sum(weights_optimal.values())
weights_optimal = {ticker: weight / total_weight for ticker, weight in weights_optimal.items()}

# Exibir os pesos otimizados para os ativos
weights_optimal_series = pd.Series(weights_optimal, index=selected_assets)
st.write("Pesos otimizados para cada ativo:")
st.write(weights_optimal_series)

# Calcular o desempenho esperado da carteira
performance = ef.portfolio_performance()
st.write(f"Retorno esperado da carteira: {performance[0]:.2f}%")
st.write(f"Risco (Desvio Padrão): {performance[1]:.2f}%")
st.write(f"Índice de Sharpe: {performance[2]:.2f}")

# Exibir um gráfico da carteira ótima com os pesos
st.bar_chart(weights_optimal_series)
