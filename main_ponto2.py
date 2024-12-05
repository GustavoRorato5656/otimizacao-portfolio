import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from pypfopt import expected_returns, risk_models, EfficientFrontier
from scipy.optimize import minimize

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

# Função de otimização do Índice de Sharpe com restrição no número de ativos
def optimize_portfolio(mean_returns, cov_matrix, num_assets):
    # Número de ativos
    num_total_assets = len(mean_returns)
    
    # Função objetivo: Maximizar o Índice de Sharpe
    def objective(weights):
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -portfolio_return / portfolio_volatility  # Maximizar Sharpe, que é retorno/risco (volatilidade)
    
    # Restrição: soma dos pesos deve ser igual a 1
    def constraint(weights):
        return np.sum(weights) - 1
    
    # Restrição: O número de ativos escolhidos deve ser igual a 'num_assets'
    def asset_count_constraint(weights):
        return np.count_nonzero(weights) - num_assets
    
    # Inicialização dos pesos: começamos com pesos iguais
    initial_weights = np.ones(num_total_assets) / num_total_assets
    
    # Definindo as restrições e os limites (pesos entre 0 e 1)
    constraints = [{'type': 'eq', 'fun': constraint}, {'type': 'eq', 'fun': asset_count_constraint}]
    bounds = [(0, 1)] * num_total_assets
    
    # Otimização
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if result.success:
        return result.x
    else:
        raise Exception("Otimização falhou. Tente novamente.")

# Rodar a otimização para encontrar a melhor carteira com 'num_assets' ativos
optimized_weights = optimize_portfolio(mean_returns, cov_matrix, num_assets)

# Criar um DataFrame com os resultados dos pesos
optimized_weights_df = pd.Series(optimized_weights, index=mean_returns.index)

# Filtrar os ativos que têm peso maior que 0 (para que o portfólio tenha exatamente 'num_assets' ativos)
optimized_weights_df = optimized_weights_df[optimized_weights_df > 0]

# Exibir os ativos selecionados e os pesos
st.write(f"Ativos selecionados para a carteira ({num_assets} ativos):")
st.write(optimized_weights_df)

# Calcular o desempenho esperado da carteira
ef = EfficientFrontier(mean_returns[optimized_weights_df.index], cov_matrix.loc[optimized_weights_df.index, optimized_weights_df.index])
ef.set_weights(dict(zip(optimized_weights_df.index, optimized_weights_df.values)))

performance = ef.portfolio_performance()
st.write(f"Retorno esperado: {performance[0]:.2f}%")
st.write(f"Risco (Desvio Padrão): {performance[1]:.2f}%")
st.write(f"Índice de Sharpe: {performance[2]:.2f}")

# Exibir um gráfico da carteira ótima
st.bar_chart(optimized_weights_df)
