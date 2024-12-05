import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from pypfopt import expected_returns, risk_models, EfficientFrontier

# Defina o universo de ativos por grupo
assets_universe = {
    'AAPL': ('Apple', 'ações', 'americano'), 'MSFT': ('Microsoft', 'ações', 'americano'), 'TSLA': ('Tesla', 'ações', 'americano'),
    'GOOG': ('Alphabet', 'ações', 'americano'), 'AMZN': ('Amazon', 'ações', 'americano'), 'SPY': ('S&P 500 ETF', 'ações', 'americano'),
    'BTC-USD': ('Bitcoin', 'criptomoedas', ''), 'ETH-USD': ('Ethereum', 'criptomoedas', ''),
    'NVDA': ('NVIDIA', 'ações', 'americano'), 'META': ('Meta', 'ações', 'americano')
}

# Grupos definidos (considerando que a entrada de cada ativo tem o nome, o tipo e a origem)
cryptos = ['BTC-USD', 'ETH-USD']
stocks_usa = ['AAPL', 'MSFT', 'TSLA', 'GOOG', 'AMZN', 'SPY', 'NVDA', 'META']
# Os outros grupos (renda fixa, ações brasileiras, etc.) precisam ser adicionados ao universo conforme necessário.

# Entrada do usuário para o número de ativos na carteira
num_assets = st.number_input("Quantos ativos você deseja na sua carteira?", min_value=1, max_value=10, value=3)

# Baixar os dados históricos para todos os ativos do universo
st.write("Baixando dados históricos dos ativos...")

# Baixar os dados históricos (ajustados) de todos os ativos do universo
data = yf.download(list(assets_universe.keys()), start="2020-01-01", end="2024-01-01")['Adj Close']

# Calcular os retornos esperados e a matriz de covariância
mean_returns = expected_returns.mean_historical_return(data)
cov_matrix = risk_models.sample_cov(data)

# Função para otimizar a carteira com a restrição de ter exatamente 2 ativos por grupo
def optimize_portfolio(mean_returns, cov_matrix, num_assets):
    # Defina os grupos de ativos
    groups = {
        'criptomoedas': cryptos,
        'ações_americanas': stocks_usa,
        # Adicione outros grupos aqui, como ações brasileiras e de outros mercados
    }

    selected_assets = []

    # Selecionar 2 ativos de cada grupo, se disponível
    for group, tickers in groups.items():
        tickers_in_group = mean_returns[tickers].index  # ativos do grupo
        if len(tickers_in_group) > 2:
            selected_assets.extend(tickers_in_group[:2])  # Selecionar 2 ativos

    # Verifique se temos ativos suficientes
    if len(selected_assets) < num_assets:
        raise ValueError(f"Não há ativos suficientes para selecionar {num_assets} ativos com 2 ativos por grupo.")

    # Agora que temos os ativos selecionados, calcule a carteira ótima com esses ativos
    selected_data = data[selected_assets]
    selected_mean_returns = mean_returns[selected_assets]
    selected_cov_matrix = cov_matrix.loc[selected_assets, selected_assets]

    # Crie um objeto EfficientFrontier com os ativos selecionados
    ef = EfficientFrontier(selected_mean_returns, selected_cov_matrix)
    
    # Maximizar o Índice de Sharpe
    weights = ef.max_sharpe()

    # Ordenar os ativos por peso
    sorted_weights = pd.Series(weights).sort_values(ascending=False)

    # Selecionar os 'num_assets' melhores ativos com base nos pesos
    selected_assets_weights = sorted_weights.head(num_assets)

    # Normalizar os pesos para garantir que a soma seja 1
    total_weight = selected_assets_weights.sum()
    normalized_weights = selected_assets_weights / total_weight

    return normalized_weights

# Executar a otimização
try:
    optimized_weights = optimize_portfolio(mean_returns, cov_matrix, num_assets)

    # Exibir os ativos selecionados e os pesos
    st.write(f"Ativos selecionados para a carteira ({num_assets} ativos):")
    st.write(optimized_weights)

    # Criar um novo objeto Efficient Frontier apenas com os ativos selecionados
    ef = EfficientFrontier(mean_returns[optimized_weights.index], cov_matrix.loc[optimized_weights.index, optimized_weights.index])
    ef.set_weights(dict(zip(optimized_weights.index, optimized_weights.values())))

    # Calcular o desempenho esperado da carteira
    performance = ef.portfolio_performance()
    st.write(f"Retorno esperado: {performance[0]:.2f}%")
    st.write(f"Risco (Desvio Padrão): {performance[1]:.2f}%")
    st.write(f"Índice de Sharpe: {performance[2]:.2f}")

    # Exibir um gráfico da carteira ótima com os pesos normalizados
    st.bar_chart(normalized_weights)

except Exception as e:
    st.error(f"Erro: {str(e)}")
