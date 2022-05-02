import json

class Cargar_datos_metadata:

    """constructor"""
    def __init__(self,ruta):
        self.ruta=ruta

    def cargar_jupyter_metadata(ruta):
        """
        Método que extrae las celdas de metadatos de un notebook
        Returns: Lista
        Devuelve una lista con las celdas de metadatos del notebook.
        """
        #Cargo el archivo en formato JSON
        file=open(ruta,'r',encoding='utf-8')
        jmain=json.load(file)
        cadena_metadata =[]

        #Recorro el archivo , especificamente las celdas de tipo metadata
        for i in jmain['metadata']:
            #Si es autor
            if(i=='authors'):
             cadena_metadata.append(jmain['metadata'][i])
            #Si es un titulo
            if (i=='title'):
                cadena_metadata.append(jmain['metadata'][i])

        file.close()
        return cadena_metadata;
