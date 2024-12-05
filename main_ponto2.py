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

# Maximizar o Índice de Sharpe (com restrição de número de ativos)
# Primeiramente, calcular a carteira ótima com todos os ativos
weights_all_assets = ef.max_sharpe()

# Ordenar os ativos por peso e escolher os 'num_assets' mais relevantes
sorted_weights = pd.Series(weights_all_assets).sort_values(ascending=False)

# Selecionar apenas os 'num_assets' melhores ativos com base no peso
selected_assets = sorted_weights.head(num_assets)

# Normalizar os pesos para garantir que a soma seja 1
total_weight = selected_assets.sum()
normalized_weights = selected_assets / total_weight

# Filtrar ativos que têm peso menor que 0.0001
normalized_weights = normalized_weights[normalized_weights >= 0.0001]

# Se houver ativos suficientes para compor a carteira
if len(normalized_weights) >= num_assets:  # Garantir que tenhamos pelo menos 'num_assets' ativos
    # Exibir os ativos selecionados e os pesos normalizados
    st.write(f"Ativos selecionados para a carteira ({len(normalized_weights)} ativos):")
    st.write(normalized_weights)

    # Recalcular o desempenho esperado da carteira com os pesos normalizados
    ef = EfficientFrontier(mean_returns[normalized_weights.index], cov_matrix.loc[normalized_weights.index, normalized_weights.index])
    ef_weights = normalized_weights.values  # Pesos normalizados
    ef.set_weights(dict(zip(normalized_weights.index, ef_weights)))  # Definir pesos para o EfficientFrontier

    # Calcular o desempenho esperado da carteira
    performance = ef.portfolio_performance()
    st.write(f"Retorno esperado: {performance[0]:.2f}%")
    st.write(f"Risco (Desvio Padrão): {performance[1]:.2f}%")
    st.write(f"Índice de Sharpe: {performance[2]:.2f}")

    # Exibir um gráfico da carteira ótima com os pesos normalizados
    st.bar_chart(normalized_weights)
else:
    st.write(f"Não há ativos suficientes com peso significativo para formar uma carteira de {num_assets} ativos.")
