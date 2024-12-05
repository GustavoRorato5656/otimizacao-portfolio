import streamlit as st
import yfinance as yf

# Título e descrição inicial
st.title("Otimização de Portfólio")
st.write("Bem-vindo ao aplicativo de otimização de portfólio usando Streamlit!")

# Entrada do usuário: tickers de ativos
tickers = st.text_input("Digite os tickers dos ativos separados por vírgula:", "AAPL, MSFT, TSLA")
st.write("Você selecionou os seguintes ativos:", tickers)

# Botão para buscar dados e calcular a carteira
if st.button("Clique para calcular a carteira ótima"):
    st.write("Buscando dados e calculando a carteira...")

    # Baixar dados históricos dos ativos
    try:
        tickers_list = tickers.split(",")  # Converte os tickers digitados em uma lista
        data = yf.download(tickers_list, period="1y")  # Passa a lista de tickers
        st.write("Preços históricos dos ativos selecionados:")
        st.line_chart(data['Close'])  # Gráfico dos preços de fechamento
    except Exception as e:
        st.write("Erro ao buscar os dados:", e)
