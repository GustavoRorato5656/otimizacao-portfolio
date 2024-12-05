import streamlit as st
import yfinance as yf

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

    except Exception as e:
        st.error(f"Ocorreu um erro ao baixar os dados: {str(e)}")
