import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from pypfopt import expected_returns, risk_models, EfficientFrontier

# Defina o universo de ativos
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

# Função para otimizar a carteira com um número específico de ativos
def optimize_portfolio(mean_returns, cov_matrix, num_assets):
    try:
        # Maximizar o Índice de Sharpe
        ef = EfficientFrontier(mean_returns, cov_matrix)
        weights = ef.max_sharpe()

        # Ordenar os ativos por peso
        sorted_weights = pd.Series(weights).sort_values(ascending=False)

        # Selecionar os 'num_assets' melhores ativos com base nos pesos
        selected_assets = sorted_weights.head(num_assets)

        # Verificar se a soma dos pesos é igual a 1 e normalizar se necessário
        total_weight = selected_assets.sum()
        if total_weight != 1:
            selected_assets = selected_assets / total_weight

        return selected_assets

    except Exception as e:
        raise Exception("Otimização falhou. Tente novamente.") from e

# Executar a otimização
try:
    optimized_weights = optimize_portfolio(mean_returns, cov_matrix, num_assets)

    # Exibir os ativos selecionados e os pesos
    st.write(f"Ativos selecionados para a carteira ({num_assets} ativos):")
    st.write(optimized_weights)

    # Criar um novo objeto Efficient Frontier apenas com os ativos selecionados
    ef.set_weights(dict(zip(optimized_weights.index, optimized_weights.values())))

    # Calcular o desempenho esperado da carteira
    performance = ef.portfolio_performance()
    st.write(f"Retorno esperado: {performance[0]:.2f}%")
    st.write(f"Risco (Desvio Padrão): {performance[1]:.2f}%")
    st.write(f"Índice de Sharpe: {performance[2]:.2f}")

    # Exibir um gráfico da carteira ótima
    st.bar_chart(optimized_weights)

except Exception as e:
    st.error(str(e))
