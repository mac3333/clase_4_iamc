"""
palabras reservadas python

['False', 'None', 'True', '__peg_parser__', 'and', 'as', 'assert', 'async', 'await', 'break', 'class', 'continue',
'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda',
'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield']

"""

# variables....
esto_es_una_variable = 12
esto_es_una_variable_2 = "cadenas de texto"
esto_es_una_variable_extra = 22.3
# tipos de datos
esto_es_un_str = str()
esto_es_un_float = float()
esto_es_un_int = int()

# estruturas de datos
esto_es_una_variable_3 = []
esto_es_una_variable_4 = list()
esto_es_una_variable_5 = {}
esto_es_una_variable_6 = dict()
esto_es_una_variable_7 = ()
esto_es_una_variable_8 = tuple()

# ingresar datos
# input()
# ingrese_datos = input()
# print("tipo de dato", type(ingrese_datos))
# # usar conversiones de datos
# ingrese_datos = int(ingrese_datos)
# print("tipo de dato con conversion", type(ingrese_datos))
# ingrese_datos = float(ingrese_datos)
# print("tipo de dato con float", type(ingrese_datos))
# print("tipo de dato con float, {}".format(type(ingrese_datos)))


# bid_book_price = [22.2, 22.1, 22, 18.23, 10]
#
# for cada_elemento in bid_book_price:
#     print(cada_elemento)
#     division = cada_elemento / 2
#     print(division)

bid_book_price = []   # esto es una lista vacia
lista_de_precios_de_fix = [22.2, 22.1, 22, 18.23, 10]
# print("lista bid_book_price vacia: {}".format(bid_book_price))  # recomiendo esto, me gusta mas y es mejor en pep8

for elemento in lista_de_precios_de_fix:
    bid_book_price.append(elemento)  # lleno la lista con el append
    if elemento == 22.1:
        # print(elemento)
        pass
    else:
        pass
        # print("no es 22.1")

# print("lista bid_book_price llena", bid_book_price)


diccionario = {"bid_price": bid_book_price, "ask_price": [24.2, 23.1, 22, 17.23, 8]}

punta_bid = diccionario["bid_price"][0]
punta_ask = diccionario["ask_price"][0]
#  print(diccionario["bid_price"][0] / diccionario["ask_price"][0])


