import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pypfopt import expected_returns, risk_models, EfficientFrontier

# Título do aplicativo
st.title("Otimização de Portfólio com Sharpe")
st.write("Bem-vindo ao aplicativo de otimização de portfólio usando Streamlit!")

# 1. Entrada de dados: seleção de tickers e horizonte de tempo
tickers = st.text_input("Digite os tickers dos ativos separados por vírgula:", "AAPL, MSFT, TSLA")
start_date = st.date_input("Data de Início", pd.to_datetime("2020-01-01"))
end_date = st.date_input("Data de Fim", pd.to_datetime("2024-01-01"))

# 2. Buscar os dados históricos
if st.button("Calcular a Carteira Ótima"):
    st.write("Buscando dados e calculando a carteira...")

    # Baixar os dados históricos usando yfinance
    tickers_list = tickers.split(",")  # Converte os tickers para lista
    data = yf.download(tickers_list, start=start_date, end=end_date)['Adj Close']

    # Exibir os dados históricos
    st.write("Dados históricos de preços de fechamento ajustados:")
    st.dataframe(data)

    # 3. Cálculo dos retornos esperados e matriz de covariância
    returns = expected_returns.mean_historical_return(data)
    cov_matrix = risk_models.sample_cov(data)

    # 4. Otimização do portfólio usando PyPortfolioOpt
    ef = EfficientFrontier(returns, cov_matrix)
    weights = ef.max_sharpe()  # Maximizar o Índice de Sharpe

    # Resultados
    st.write("Pesos ótimos para cada ativo com o objetivo de maximizar o Índice de Sharpe:")
    st.write(weights)

    # Cálculo da performance do portfólio otimizado
    performance = ef.portfolio_performance(verbose=True)
    st.write(f"Retorno esperado: {performance[0]:.2f}")
    st.write(f"Volatilidade esperada: {performance[1]:.2f}")
    st.write(f"Índice de Sharpe: {performance[2]:.2f}")

    # Exibindo gráfico da composição do portfólio
    st.write("Composição do portfólio:")
    st.bar_chart(weights)

    # 5. Sugestão de uma carteira com número específico de ativos
    num_assets = st.number_input("Número de ativos na carteira:", min_value=1, max_value=len(tickers_list), value=3)
    st.write(f"Calculando a melhor carteira com {num_assets} ativos...")

    # Gerando combinações possíveis de ativos
    from itertools import combinations
    asset_combinations = list(combinations(tickers_list, num_assets))

    # Calculando a carteira ótima para cada combinação
    best_sharpe = -np.inf
    best_combination = None
    best_weights = None

    for combination in asset_combinations:
        subset_data = data[list(combination)]
        subset_returns = expected_returns.mean_historical_return(subset_data)
        subset_cov_matrix = risk_models.sample_cov(subset_data)
        
        ef_subset = EfficientFrontier(subset_returns, subset_cov_matrix)
        subset_weights = ef_subset.max_sharpe()
        subset_performance = ef_subset.portfolio_performance()

        if subset_performance[2] > best_sharpe:  # Comparar o Índice de Sharpe
            best_sharpe = subset_performance[2]
            best_combination = combination
            best_weights = subset_weights

    # Exibindo a melhor combinação
    st.write(f"A melhor carteira com {num_assets} ativos é composta por:")
    st.write(best_combination)
    st.write("Pesos dessa carteira:")
    st.write(best_weights)

    # Exibindo gráfico da carteira ótima
    st.write(f"Composição da carteira ótima com {num_assets} ativos:")
    st.bar_chart(best_weights)
