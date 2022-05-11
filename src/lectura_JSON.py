
import json

class Lectura_JSON:

        def __init__(self,ruta):
            self.ruta=ruta

        def obtener_dependencias(ruta):
            """
            Método encargado de leer los archivos de la ruta especificada para extraer los dependencias del notebook
            Returns:
            Una lista con las dependencias del notebook
            """
            try:
                requerimientos2=[]
                with open(ruta, "r") as j:
                    datos=json.load(j)
                    if 'requirements' in datos:
                        requerimientos = datos['requirements']
                        for i in requerimientos:
                                requerimientos2.append({'name': i, 'version': requerimientos[i]})
                    else:
                        posicion=ruta.find("directory")
                        posicion_nombre=ruta.find('/tmp/')
                        cadena_sin_nombre=ruta[:posicion]
                        nombre=cadena_sin_nombre[len('/tmp/'):-1]
                        clave=cadena_sin_nombre+nombre
                        for i in datos[clave]:
                           requerimientos= i['dependencies']
                        for i in requerimientos:
                            requerimientos2.append({'name':i['import'], 'type':i['type']})
                return requerimientos2
            except:
                print("No se han podido obtener las dependencias mediante inspect4py")

        def obtener_descripcion(ruta):
            """
            Método encargado de extraer la descripcion del archivo que se especifica en la ruta
            Returns:
            Una cadena que contiene la descripción en caso de poder ser extraida.
            """
            try:
                with open(ruta, "r") as j:
                 datos=json.load(j)
                return datos['file']['doc']['short_description']
            except:
                print("No se han podido obtener la descripcion mediante inspect4py")
                return []


        def obtener_llamadas_funciones(ruta):
            """
            Método encargado de extraer las llamadas a funciones de un archivo especificado en la ruta de entrada
            Returns:
            Una lista con las llamadas a las funciones.
            """
            try:
                with open(ruta, "r") as j:
                    datos=json.load(j)
                return datos['body']
            except:
                print("NO se han podido obtener las llamadas a funciones mediante inspect4py")
