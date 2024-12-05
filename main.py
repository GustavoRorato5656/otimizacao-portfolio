import streamlit as st
import yfinance as yf
import pandas as pd
from pypfopt import EfficientFrontier, risk_models, expected_returns
import numpy as np

# Universo de ativos pré-definido (exemplo com ações e criptomoedas)
universo_ativos = [
    "AAPL", "MSFT", "TSLA", "GOOG", "AMZN", "META", # Ações de empresas
    "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD" # Criptomoedas
]

# Entrada do usuário para os tickers dos ativos (pode ser manual ou pré-definido)
tickers = st.text_input("Digite os tickers dos ativos separados por vírgula:", "AAPL, MSFT, TSLA")

# Entrada do usuário para o número de ativos na carteira
num_ativos = st.number_input("Quantos ativos você deseja na carteira?", min_value=1, max_value=len(universo_ativos), value=3)

# Se o campo de tickers não estiver vazio
if tickers:
    # Converter a string de tickers em uma lista
    tickers_list = [ticker.strip() for ticker in tickers.split(",")]
    
    # Verificar se o número de ativos selecionado é menor ou igual ao número de tickers disponíveis
    if num_ativos <= len(tickers_list):
        try:
            # Baixar os dados históricos de preços ajustados
            data = yf.download(tickers_list, start="2020-01-01", end="2024-01-01")['Adj Close']
            
            # Exibir os dados
            st.write("Dados históricos de preços de fechamento ajustados:")
            st.dataframe(data)

            # Exibir o gráfico dos dados
            st.line_chart(data)

            # Calcular os retornos diários
            returns = data.pct_change().dropna()

            # Calcular os retornos esperados e a matriz de covariância
            mean_returns = expected_returns.mean_historical_return(data)
            cov_matrix = risk_models.sample_cov(data)

            # Inicializar o otimizador de portfólio
            ef = EfficientFrontier(mean_returns, cov_matrix)

            # Calcular os melhores pesos com base no Índice de Sharpe
            weights = ef.max_sharpe()

            # Exibir os pesos dos ativos na carteira ótima
            st.write("Pesos dos ativos na carteira ótima (maximização do Índice de Sharpe):")
            st.dataframe(pd.Series(weights).sort_values(ascending=False))

            # Exibir o desempenho esperado da carteira ótima
            performance = ef.portfolio_performance(verbose=True)
            st.write(f"Desempenho esperado da carteira ótima: Retorno: {performance[0]*100:.2f}% | Volatilidade: {performance[1]*100:.2f}% | Índice de Sharpe: {performance[2]:.2f}")

        except Exception as e:
            st.error(f"Ocorreu um erro ao baixar os dados: {str(e)}")

    else:
        st.error(f"O número de ativos selecionado ({num_ativos}) é maior do que os tickers inseridos ({len(tickers_list)}). Escolha um número menor ou igual.")
    
    # Sugestão de carteira ótima com um número específico de ativos
    st.write("Buscando a melhor carteira com número específico de ativos...")
    
    # Gerar o universo de ativos completo para sugestão de carteira
    try:
        # Baixar os dados históricos do universo de ativos
        data_universo = yf.download(universo_ativos, start="2020-01-01", end="2024-01-01")['Adj Close']

        # Calcular os retornos diários
        returns_universo = data_universo.pct_change().dropna()

        # Calcular os retornos esperados e a matriz de covariância
        mean_returns_universo = expected_returns.mean_historical_return(data_universo)
        cov_matrix_universo = risk_models.sample_cov(data_universo)

        # Inicializar o otimizador de portfólio
        ef_universo = EfficientFrontier(mean_returns_universo, cov_matrix_universo)

        # Otimizar a carteira para o número de ativos desejado
        ef_universo.set_weights(weights)
        
        # Exibir a carteira sugerida
        carteira_sugerida = ef_universo.portfolio_performance(verbose=True)
        
        st.write(carteira_sugerida)
    except Exception as e:
        st.error(f"Ocorreu um erro ao baixar os dados do universo de ativos: {str(e)}")
