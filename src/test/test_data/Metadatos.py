# %%
"""
Ejemplo de una imagen
"""

# %%
"""
![image.png](attachment:image.png)
"""

# %%
"""
Ejemplo de una tabla
"""

# %%
import pandas as pd
#creacion de una tabla
tabla=pd.DataFrame(data= [[1,1],[2,4],[3,9]],
                    columns = ['numero', 'cuadrado'])
print(tabla)

# %%
"""
Ejemplo de grafico
"""

# %%
import matplotlib.pyplot as plt

x= [1,2,3,4,5]
y=[1,4,9,16,25]

plt.plot(x,y)
plt.show()

# %%
import csv

with open('example.csv', newline='') as File:  
    reader = csv.reader(File)
    for row in reader:
        print(row)