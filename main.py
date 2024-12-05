import streamlit as st
import yfinance as yf

# Título e descrição inicial
st.title("Otimização de Portfólio por Guilherme Goya e Gustavo Rorato")
st.write("Neste aplicativo de otimização de portfólio você encontrará sua carteira ótima")

# Entrada do usuário: tickers de ativos
tickers = st.text_input("Digite os tickers dos ativos escolhidos separados por vírgula:", "AAPL, MSFT, TSLA")
st.write("Você selecionou os seguintes ativos:", tickers)

# Botão para buscar dados e calcular a carteira
if st.button("Clique para calcular a carteira ótima"):
    st.write("Buscando dados e calculando a carteira...")

    # Baixar dados históricos dos ativos
    try:
        data = yf.download(tickers.split(","), period="1y")  # Dados do último ano
        st.write("Preços históricos dos ativos selecionados:")
        st.line_chart(data['Close'])  # Gráfico dos preços de fechamento
    except Exception as e:
        st.write("Erro ao buscar os dados:", e)
