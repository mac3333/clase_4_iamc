
# funciones
def nombre_de_la_funcion(nombre_de_texto):
    print(nombre_de_texto)

def otra_funcion():
    llamar_otra_funcion = nombre_de_la_funcion("estoy llamando otra funcion")
    return llamar_otra_funcion


# nombre_de_la_funcion(nombre_de_texto=1)
# nombre_de_la_funcion(nombre_de_texto=[{"numero10": 10, "numero20": 20}])
# nombre_de_la_funcion(nombre_de_texto=(11, 23))
# nombre_de_la_funcion(nombre_de_texto={"numero1": 1, "numero2": 2})
# nombre_de_la_funcion(nombre_de_texto=[1, 2, 3, 4, 5, 6])
# nombre_de_la_funcion(nombre_de_texto="esto es una funcion")
# otra_funcion()

# clases
class SerHumano:
    # aca van los atributos de la clase
    cabeza2 = None

    def __init__(self):
        self.cabeza = "el ser humano tiene una cabeza"
        self.brazos = "el ser humano tiene brazos"


# dando_vida = SerHumano()  # instanciar el objeto, en este caso, la clase SerHumano
# print(dando_vida.cabeza2)
# print(dando_vida.cabeza)


class HandlaData:

    def __init__(self, bid_price, ask_price):
        self.bid_price = bid_price
        self.ask_price = ask_price

    def get_best_price(self, price_book_name):
        """
        This method get the best price of the book name
        :param price_book_name: str param
        :return: min or max value
        """
        if price_book_name == "ask_price":
            max_value = self.ask_price[0]
            # max_value = max(self.ask_price)
            print(max_value)
            return max_value
        else:
            if price_book_name == "bid_price":
                min_value = self.bid_price[0]
                # min_value = min(self.bid_price)
                print(min_value)
                return min_value

#
# fix_byma = HandlaData(bid_price=[22, 21, 20, 19, 18], ask_price=[23, 24, 25, 26, 27])
# fix_byma.get_best_price(price_book_name="bid_price")
# fix_byma.get_best_price(price_book_name="ask_price")