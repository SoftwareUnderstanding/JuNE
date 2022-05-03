# %%
x=0
x=x+1
y=2

# %%
import numpy

# %%
"""
Ejemplo de celda de texto
"""

# %%
import csv
 
with open('example.csv', newline='') as File:  
    reader = csv.reader(File)
    for row in reader:
        print(row)