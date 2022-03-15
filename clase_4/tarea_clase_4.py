def cf_series(face, coupon, maturity, get_plot=False):
    """
    Una función que genera una lista que contiene la estructura de flujos de efectivo para un bono que paga un cupón anual
    :param get_plot: to get plot
    :param face: the final redemption value of the bond, also known as par amount or principal.
    :param coupon: the coupon rate paid by the bond.
    :param maturity: the number of years until the bond matures and pays back it's face value.
    :return: a list that contains the modelled cash flows of the bond
    """
    pass
    return


def discount_factors(spot_rates, get_plot=False):
    """
    una función que obtiene los factores de descuento. se puede utilizar las tasas libres de riesgo vigentes
    hasta el vencimiento del bono. La estructura de estas tasas se conoce comúnmente como curva de rendimiento al
    contado o estructura de plazos.
    :param spot_rates: list of rates to discount
    :param get_plot: to get plot
    :return: discount_factors
    """
    pass
    return


def bond_present_value(bond_a_cf, bond_discount_factors, get_plot=False):
    """
    encontraremos el valor presente de los flujos de efectivo del bono utilizando los factores de descuento, cuya suma
    es el valor del bono. Cuando combinamos los flujos de efectivo modelados y los factores de descuento, llegamos a la
    forma abierta del precio del bono.
    :param bond_a_cf: cash flow
    :param bond_discount_factors: discount factor
    :param get_plot: to get plots
    :return:
    """
    pass
    return
