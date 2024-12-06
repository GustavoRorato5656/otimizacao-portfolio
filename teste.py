# Função para otimização de portfólio
def optimize_portfolio(data):
    """
    Otimiza a alocação de ativos usando o Índice de Sharpe.

    Args:
        data (DataFrame): Dados históricos dos preços.

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

    # Calcular métricas de desempenho
    performance = ef.portfolio_performance(verbose=False)

    return cleaned_weights, performance


# Atualizar o dashboard
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

            # Selecionar apenas um subconjunto aleatório de ativos
            selected_tickers = np.random.choice(data.columns, size=num_assets, replace=False)
            selected_data = data[selected_tickers]

            # Otimizar portfólio
            top_weights, performance = optimize_portfolio(selected_data)

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

