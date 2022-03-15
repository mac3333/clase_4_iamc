# Ejercicios Python:


# Ejercicio de Lista

# 1. Escriba un programa en Python para sumar todos los elementos de una lista.

def sum_list(items):
    sum_numbers = 0
    for x in items:
        sum_numbers += x
    return sum_numbers
print(sum_list([1,2,-8]))


# 2. Escriba un programa en Python para multiplicar todos los elementos de una lista.

def multiply_list(items):
    tot = 1
    for x in items:
        tot *= x
    return tot
print(multiply_list([1,2,-8]))


# 3. Escriba un programa en Python para obtener el número más grande de una lista

def max_num_in_list( list ):
    max = list[ 0 ]
    for a in list:
        if a > max:
            max = a
    return max
print(max_num_in_list([1, 2, -8, 0]))


# 4. Escriba un programa en Python para obtener el número más pequeno de una lista

def smallest_num_in_list( list ):
    min = list[ 0 ]
    for a in list:
        if a < min:
            min = a
    return min
print(smallest_num_in_list([1, 2, -8, 0]))

# 5. Escriba un programa de Python para contar el número de cadenas de texto donde la longitud de la cadena es 2 o más y el primer y último carácter son los mismos de una 
# lista de cadenas dada.
# Lista de muestras: ['abc', 'xyz', 'aba', '1221']
# Resultado esperado: 2

def match_words(words):
  ctr = 0

  for word in words:
    if len(word) > 1 and word[0] == word[-1]:
      ctr += 1
  return ctr

print(match_words(['abc', 'xyz', 'aba', '1221']))

# 6. Escriba un programa de Python para obtener una lista, ordenada en orden creciente por el último elemento de cada tupla de una lista dada de tuplas no vacías. 
# Lista de muestra: [(2, 5), (1, 2), (4, 4), (2, 3), (2, 1)]
# Resultado esperado: [(2, 1), (1, 2), (2, 3), (4, 4), (2, 5)] 


def last(n): return n[-1]

def sort_list_last(tuples):
  return sorted(tuples, key=last)

print(sort_list_last([(2, 5), (1, 2), (4, 4), (2, 3), (2, 1)]))


# 7. Escriba un programa de Python para eliminar duplicados de una lista.

a = [10,20,30,20,10,50,60,40,80,50,40]

dup_items = set()
uniq_items = []
for x in a:
    if x not in dup_items:
        uniq_items.append(x)
        dup_items.add(x)

print(dup_items)


# 8. Escriba un programa de Python para verificar que una lista esté vacía o no.
l = []
if not l:
  print("List is empty")



