import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns
import streamlit as st
import pandas as pd
import numpy as np

# Definir o universo de ativos
UNIVERSO_ATIVOS = {
    "Criptomoedas": ["BTC-USD", "ETH-USD"],
    "Renda Fixa Brasil": ["IRFM11.SA", "IMAB11.SA"],
    "Ações Brasil": ["PETR4.SA", "VALE3.SA"],
    "Ações EUA": ["AAPL", "MSFT"],
    "Ações China": ["BABA", "JD"]
}

# Função para obter dados históricos
def get_data(tickers, start_date, end_date):
    """
    Obtém os preços ajustados de fechamento para uma lista de ativos.

    Args:
        tickers (list): Lista de ativos (símbolos).
        start_date (str): Data de início no formato 'YYYY-MM-DD'.
        end_date (str): Data de término no formato 'YYYY-MM-DD'.

    Returns:
        DataFrame: Dados históricos dos preços ajustados.
    """
    data = yf.download(tickers, start=start_date, end=end_date)
    return data['Adj Close']

# Função para otimização de portfólio
def optimize_portfolio(data, num_assets):
    """
    Otimiza a alocação de ativos usando o Índice de Sharpe.

    Args:
        data (DataFrame): Dados históricos dos preços.
        num_assets (int): Número de ativos na carteira.

    Returns:
        dict, tuple: Pesos ótimos dos ativos e métricas de desempenho.
    """
    # Garantir que não haja valores ausentes
    data = data.dropna()

    # Calcular retornos esperados e matriz de covariância
    mu = expected_returns.mean_historical_return(data)
    S = risk_models.sample_cov(data)
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()

    # Selecionar os ativos com os maiores pesos
    sorted_weights = sorted(cleaned_weights.items(), key=lambda x: x[1], reverse=True)
    top_assets = sorted_weights[:num_assets]
    top_weights = {asset: weight for asset, weight in top_assets}

    # Calcular métricas de desempenho
    performance = ef.portfolio_performance(verbose=False)

    return top_weights, performance

# Função para criar o dashboard
def create_dashboard():
    """
    Cria o dashboard interativo para sugestão de carteira ótima.
    """
    st.title("Dashboard de Otimização de Carteiras")
    st.markdown("""
        *Bem-vindo!*
        Este aplicativo sugere uma carteira ótima utilizando o modelo de Máximo Índice de Sharpe.
        Selecione os parâmetros abaixo e clique em "Calcular" para obter sua carteira ideal.
    """)

    # Seleção das categorias de ativos
    selected_categories = st.multiselect(
        "Escolha as categorias de ativos:",
        list(UNIVERSO_ATIVOS.keys()),
        default=list(UNIVERSO_ATIVOS.keys())
    )
    tickers = [ticker for category in selected_categories for ticker in UNIVERSO_ATIVOS[category]]

    # Seleção do número de ativos
    num_assets = st.number_input(
        "Número de ativos na carteira:",
        min_value=1,
        max_value=len(tickers),
        value=3
    )

    # Seleção do horizonte de tempo
    start_date = st.date_input("Data de início:")
    end_date = st.date_input("Data de término:")

    if start_date >= end_date:
        st.error("A data de início deve ser anterior à data de término.")
        return

    # Botão para calcular
    if st.button("Calcular"):
        try:
            # Obter dados históricos para os ativos selecionados
            data = get_data(tickers, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            if data.empty:
                st.error("Não foram encontrados dados para os ativos selecionados no período informado.")
                return

            # Otimizar portfólio
            top_weights, performance = optimize_portfolio(data, num_assets)

            # Filtrar dados para os ativos selecionados
            selected_tickers = list(top_weights.keys())
            selected_data = data[selected_tickers]

            # Exibir resultados
            st.write("### Pesos Ótimos dos Ativos:")
            st.bar_chart(pd.DataFrame.from_dict(top_weights, orient='index', columns=['Peso']))

            # Exibir métricas de desempenho
            st.write("### Desempenho da Carteira:")
            st.write(f"- *Retorno Esperado:* {performance[0]:.2%}")
            st.write(f"- *Risco (Desvio Padrão):* {performance[1]:.2%}")
            st.write(f"- *Índice de Sharpe:* {performance[2]:.2f}")
        
        except Exception as e:
            st.error(f"Ocorreu um erro: {e}")

# Execução do aplicativo
if __name__ == "__main__":
    create_dashboard()

