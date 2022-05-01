
class Depurar_datos:

    def __init__(self,cadena_datos):
        self.cadena_datos = cadena_datos


    def eliminar_espacios_saltoslinea(cadena_datos):
        """
        Método encargado de eliminar espacios o saltos de linea de una descripcion
        Returns:
        La cadena sin saltos de linea o espacios innecesarios.
        """
        #Recorro la cadena recibida
        for i in range(len(cadena_datos)):
            x=0
            while(x<(len(cadena_datos[i])-1)):
                    ##Elimino los elementos que son unicamente saltos de linea
                    if (cadena_datos[i][x] == '\n' or cadena_datos[i][x]== ''):
                      del cadena_datos[i][x]

                    else:
                        ##Elimino los espacios y saltos de linea del resto de elementos
                        if ('\n' in cadena_datos[i][x]):
                            cadena_sinsaltolinea= cadena_datos[i][x].rstrip()
                            cadena_sinespacios = cadena_sinsaltolinea.strip()
                            cadena_datos[i][x] = cadena_sinespacios
                            x = x + 1
        return cadena_datos

    def depurar_descripcion(cadena_datos):
        """
        Método encargado de detectar caracteres como saltos de linea dentro de una cadena
        Returns:
        Una lista con las cadenas que contienen caracteres invalidos
        """
        cadena_depurada=[]
        for i in cadena_datos:
            if('\n' in i or '---\n' in i):
                pass
            else:
                cadena_depurada.append(i)

        return cadena_depurada