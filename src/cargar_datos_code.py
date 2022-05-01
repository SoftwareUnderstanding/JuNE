import json

class Cargar_datos_code:

    """constructor"""
    def __init__(self,ruta):
        self.ruta=ruta

    def cargar_jupyter_code(ruta):
        """
        Método encargado de cargar las celdas de tipo código del jupyter notebook
        Returns: Lista
        Devuelve una lista que contiene las celdas de tipo code.
        """
        #Cargo el archivo en formato JSON
        file=open(ruta,'r',encoding='utf-8')
        jmain=json.load(file)
        cadena_code =[]
        #Recorro el archivo , especificamente las celdas de tipo code
        for i in jmain['cells']:
            if(i['cell_type']=='code'):
                cadena_code.append(i)
        file.close()
        return cadena_code;

    def cargar_jupyter_code_connum(ruta):
        """
        Método que extrae el código de las celdas de codigo del notebook y el número de celdas
        Returns:Diccionario
        Devuelve un diccionario que contiene por cada celda su numero de celda.
        """
        # Cargo el archivo en formato JSON
        file=open(ruta, 'r', encoding='utf-8')
        jmain = json.load(file)

        #Creo un diccionario donde almacenar la celda y el numero de celda
        codigo={}
        celdas=[]
        source=[]
        num_celdas=[]
        numero_celda=0
        # Recorro el archivo , especificamente las celdas de tipo code
        for i in jmain['cells']:
            numero_celda+=1
            if (i['cell_type'] == 'code'):
                celdas.append(i)
                num_celdas.append(numero_celda)
        #Obtengo el texto de las celdas
        for i in celdas:
            source.append(i['source'])

        codigo['celdas']=source
        codigo['num_celdas']=num_celdas

        file.close()
        return codigo;
