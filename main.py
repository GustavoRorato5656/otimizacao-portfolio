from pypfopt import EfficientFrontier, expected_returns, risk_models, objective_functions

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
    cov_matrix = risk_models.CovarianceShrinkage(data).ledoit()  # Usando regularização Ledoit

    # 4. Otimização do portfólio usando PyPortfolioOpt
    ef = EfficientFrontier(returns, cov_matrix)
    weights = ef.max_sharpe(solver="ECOS")  # Usando o solver ECOS

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
