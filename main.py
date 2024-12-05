import yfinance as yf
import pandas as pd

tickers_list = ['AAPL', 'MSFT', 'TSLA']
data = yf.download(tickers_list, start="2020-01-01", end="2024-01-01")['Adj Close']

# Exibir os dados baixados
st.write("Dados históricos de preços de fechamento ajustados:")
st.dataframe(data)
