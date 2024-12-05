import streamlit as st

st.title("Otimização de Portfólio")
st.write("Bem-vindo ao aplicativo de otimização de portfólio usando Streamlit!")
tickers = st.text_input("Digite os tickers dos ativos separados por vírgula:", "AAPL, MSFT, TSLA")
st.write("Você selecionou os seguintes ativos:", tickers)
import yfinance as yf
data = yf.download(tickers.split(","))
st.line_chart(data['Close'])
if st.button("Clique para calcular a carteira ótima"):
    st.write("Cálculo da carteira sendo feito...")

# Adicione sua lógica aqui
