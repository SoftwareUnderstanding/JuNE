import json


class Cargar_datos_markdown:

    """constructor"""
    def __init__(self,ruta):
        self.ruta=ruta

    def cargar_jupyter_markdown(ruta):
        """
        Método que extrae las celdas de tipo markdown del notebook
        Returns: Lista
        Devuelve una lista con las celdas de tipo markdown
        """
        file=open(ruta,'r',encoding='utf-8')
        #Cargo el formato en archivo JSON
        jmain=json.load(file)
        cadena_markdown=[]

        #Recorro las celdas, especificamente las de tipo markdown
        for i in jmain['cells']:
            if(i['cell_type']=='markdown'):
                cadena_markdown.append(i)

        file.close()
        return cadena_markdown ;

