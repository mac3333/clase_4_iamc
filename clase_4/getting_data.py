import pandas as pd
import yfinance as yf
import os


class YahooFinanceData:

    def __init__(self, from_period, to_period, data_period):
        self.df = pd.DataFrame()
        self.from_period = from_period
        self.to_period = to_period
        self.data_period = data_period
        self.ticker_fail = []

    def getting_yahoo_data(self, ric):
        try:
            data = yf.download(ric,
                               start=self.from_period,
                               end=self.to_period).resample(self.data_period).last().dropna()
            data.rename(columns={"Adj Close": ric}, inplace=True)
            data.drop(["Open", "High", "Low", "Close", "Volume"], axis=1, inplace=True)
            self.df = self.df.merge(data, right_index=True, left_index=True, how="outer")

            return self.df
        except Exception as e:
            self.ticker_fail.append(ric)
            print(e, self.ticker_fail)

    def reading_csv_file(self, file_name_with_extension):
        try:
            cwd = os.getcwd()  # Get the current working directory (cwd)
            file = cwd + file_name_with_extension
            data = pd.read_csv(file)  # FileNotFoundError
            print(data)
            return data
        except FileNotFoundError:
            print("El archivo no esta en el escritorio de trabajo")  # no se imprime en panta, se hace login
