import datetime as dt
import pandas as pd
from pandas_datareader import data as pdr

# Dow Jones 30
symbols_table = pd.read_html("https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average#Components",
                             header=0)[1]

symbols = list(symbols_table.loc[:, "Symbol"])
index_symbol = ['^DJI']

# Dates
start_date = '2008-01-01'
end_date = '2017-12-31'

# Download the data
data = pd.DataFrame()

# Clean all symbol labels and remove unavailable ones
for i in range(len(symbols)):
    symbols[i] = symbols[i].replace(u'\xa0', u'').replace("NYSE:", "")

symbols.remove('DOW')  # DOW data are unavailable on yahoo

for i in range(len(symbols)):
    print('Downloading.... ', i, symbols[i])

    # User pandas_reader.data.DataReader to load the desired data. As simple as that.
    data[symbols[i]] = pdr.DataReader(symbols[i], "yahoo", start_date, end_date)['Adj Close']
    data_index = pdr.DataReader(index_symbol, "yahoo", start_date, end_date)['Adj Close']

# Remove missing data from the dataframe
data = data.dropna()
data_index = data_index.dropna()

# Save the data
data.to_csv('dj30_10y.csv', sep=',', encoding='utf-8')
data_index.to_csv('dj30_index_10y.csv', sep=',', encoding='utf-8')

print(data.head())
