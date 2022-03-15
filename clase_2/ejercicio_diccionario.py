# Ejercicios Python

# Ejercicios Diccionarios


# 1. Escriba un script de Python para agregar una clave a un diccionario.

d = {0:10, 1:20}
print(d)
d.update({2:30})
print(d)

# 2. Escriba un script de Python para concatenar los siguientes diccionarios para crear uno nuevo.
# Diccionario de muestra
# dic1={1:10, 2:20}
# dic2={3:30, 4:40}
# dic3={5:50,6:60}

dic1={1:10, 2:20}
dic2={3:30, 4:40}
dic3={5:50,6:60}
dic4 = {}
for d in (dic1, dic2, dic3): dic4.update(d)
print(dic4)

# 3. Escriba un script de Python para verificar si una clave dada ya existe en un diccionario

d = {1: 10, 2: 20, 3: 30, 4: 40, 5: 50, 6: 60}
def is_key_present(x):
  if x in d:
      print('Key is present in the dictionary')
  else:
      print('Key is not present in the dictionary')
is_key_present(5)
is_key_present(9)


# 4. Escriba un programa de Python para iterar sobre diccionarios usando bucles for

d = {'x': 10, 'y': 20, 'z': 30} 
for dict_key, dict_value in d.items():
    print(dict_key,'->',dict_value)


# 5. Escriba un script de Python para generar e imprimir un diccionario que contenga un número (entre 1 y n) en la forma (x, x*x).

# Diccionario de muestra (n = 5):
# Salida esperada: {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}

n = int(input("Input a number "))
d = dict()  # d = {}

for x in range(1, n+1):
    d[x] = x*x

print(d) 


# 6. Escriba un script de Python para fusionar dos diccionarios de Python
d1 = {'a': 100, 'b': 200}
d2 = {'x': 300, 'y': 200}
d = d1.copy()
d.update(d2)
print(d)


# 7. Escriba un programa de Python para iterar sobre diccionarios usando bucles for.
d = {'Red': 1, 'Green': 2, 'Blue': 3} 
for color_key, value in d.items():
     print(color_key, 'corresponds to ', d[color_key]) 


# 8. Escriba un programa en Python para sumar todos los elementos de un diccionario.

my_dict = {'data1':100,'data2':-54,'data3':247}
print(sum(my_dict.values()))


# 9. Escriba un programa en Python para multiplicar todos los elementos de un diccionario
my_dict = {'data1':100,'data2':-54,'data3':247}
result=1
for key in my_dict:    
    result=result * my_dict[key]

print(result)


# 10. Escriba un programa Python para eliminar una clave de un diccionario.
myDict = {'a':1,'b':2,'c':3,'d':4}
print(myDict)
if 'a' in myDict: 
    del myDict['a']
print(myDict)


