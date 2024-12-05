import streamlit as st
import yfinance as yf
import pandas as pd
from pypfopt import EfficientFrontier, risk_models, expected_returns
st.title("Otimização de Portfólio com Ativos Selecionados")
st.text("By: Guilherme Goya e Gustavo Rorato")
# Entrada do usuário para os tickers dos ativos
tickers = st.text_input("Digite os tickers dos ativos separados por vírgula:", "AAPL, MSFT, TSLA")

# Se o campo de tickers não estiver vazio
if tickers:
    # Converter a string de tickers em uma lista
    tickers_list = [ticker.strip() for ticker in tickers.split(",")]
    
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
