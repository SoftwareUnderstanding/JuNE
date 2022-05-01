import pandas as pd

class Depurar_datos_outputs:

    def __init__(self, cadena_datos):
        self.cadena_datos = cadena_datos

    def eliminar_saltos_linea(cadena_datos):
        """
        Método para recorrer una cadena y eliminar los saltos de linea y espacios innecesarios
        Returns:
        Devuelve la cadena sin saltos de linea
        """
        lista_devolver=[]

        #Recorro la cadena recibida
        for i in range(len(cadena_datos)):
            for j in range(len(cadena_datos[i])):
                for elem in range(len(cadena_datos[i][j])):
                    #Dado que sigue una estructura , el primer elemento será name
                    if(elem==0):
                        diccionario='name'
                        resultado=cadena_datos[i][j].get(diccionario)
                        lista_devolver.append({diccionario : resultado})
                    #El segundo elemento corresponde a output_type
                    elif (elem==1):
                        diccionario='output_type'
                        resultado =cadena_datos[i][j].get(diccionario)
                        lista_devolver.append({diccionario : resultado})
                    #El tercer elemento corresponde a text
                    elif (elem==2):
                        diccionario='text'
                        resultado= cadena_datos[i][j].get(diccionario)
                        ## Llamo al método arreglar_cadenas para  eliminar los saltos de linea y los espacios en blanco
                        resultado_final=Depurar_datos_outputs.arreglar_cadenas(resultado)
                        lista_devolver.append({diccionario :resultado_final})

        return lista_devolver

    def arreglar_cadenas(cadena_con_saltos):
        """
        Método encargado de eliminar los saltos de linea y espacios
        Returns:
        La cadena pasada como parámetro sin saltos de linea y espacios innecesarios.
        """
        cadena_conformato=[]
        #Recorro la cadena
        for i in range(len(cadena_con_saltos)-1):
            #Elimino los saltos de linea del final
            resultado=cadena_con_saltos[i].rstrip()
            #Elimino los espacios
            resultado2=resultado.strip()
            cadena_conformato.append(resultado2)
        return cadena_conformato
