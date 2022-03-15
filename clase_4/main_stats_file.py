from getting_data import YahooFinanceData

benchmark = ["^GSPC"]
rics = ['GGAL', 'BMA', 'SUPV', 'MELI']


yahoo_data = YahooFinanceData(from_period="2001-01-01", to_period="2022-01-01", data_period="1d")

for activo in rics:
    yahoo_data.getting_yahoo_data(activo)



# yahoo_data.reading_csv_file("/prices.csv")
