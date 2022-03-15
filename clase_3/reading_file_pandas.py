import pandas as pd


class ReadFile:

    def __init__(self, path):
        self.path = path

    def read_excel_data(self):
        file = pd.read_excel(self.path)
        print(file)
        return file


file = ReadFile("/Users/nlazardim/Documents/IAMC/codigo/archivos/excel/{}".format("listado_alumnos.xlsx"))

file.read_excel_data()
