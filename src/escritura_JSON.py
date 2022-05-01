import json
import ntpath
import os
import subprocess

class Escritura_JSON:

    def crear_carpeta_JSON(ruta_input):
        try:
            cmd = 'mkdir' + ' ' + ruta_input
            print("cmd: %s" % cmd)
            proc = subprocess.Popen(cmd.encode('utf-8'), shell=True, stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = proc.communicate()
        except:
            print("No se pudo crear la carpeta con la ruta "+ ruta_input)
        return 0

    def escribir_JSON(ruta_destino,bash,autor,titulo,requerimientos,descripcion,llamadas,cadena_paths,nombre_archivo,
                      visualizaciones,imports,resultados):
        """
        Método encargado de escribir el .json con toda la información extraida
        Args:
            bash: Lista que contiene todas las lineas bash del notebook
            autor: Autor del notebook
            titulo: Titulo del notebook
            requerimientos: Lista con los requerimientos extraidos del notebook
            descripcion: Descripción del notebook
            llamadas: Llamadas que se realizan a funciones en el notebook
            cadena_paths: Conjunto de input_paths que hay en el notebook
            nombre_archivo: Nombre del notebook
            visualizaciones: Diccionario con las celdas clasificadas como visualizaciones y
            sus numeros de celdas correspondientes
            imports: Diccionario con las celdas clasificadas como imports y
            sus numeros de celdas correspondientes
            resultados: Diccionario con las celdas clasificadas obtenido por el modelo preentrenado.

        Returns:0
        """

        #Creo un diccionario donde se van a almacenar los datos
        data={}
        data['name']=nombre_archivo
        #Compruebo si los campos estan rellenos
        if bash:
            data['bash'] = bash
        if autor:
            data['author'] = autor
        if titulo:
            data['title'] = titulo
        if requerimientos:
            data['requirements']=requerimientos
        if llamadas:
            data['body']=llamadas
        if descripcion:
            if(isinstance(descripcion,list)):
                data['description'] = descripcion[0]
            else:
                data['description'] = descripcion
        if cadena_paths:
            data['paths'] = cadena_paths

        lista= str(resultados['celdas_visualizaciones'])


        #Introduzco los datos del clasificador de celda.
        if(visualizaciones!=0 or imports !=0):
            data2={}
            data_visualizaciones={}
            data_config={}
            data_procesado={}
            data_visualizaciones['Cells_number']=str(resultados['celdas_visualizaciones'])
            data_visualizaciones['Total_visualizations']=resultados['visualizaciones']
            data_config['Cells_number'] = str(resultados['celdas_configuracion'])
            data_config['Total_configurations'] = resultados['configuracion']
            data_procesado['Cells_number'] = str(resultados['celdas_procesado'])
            data_procesado['Total_processes'] = resultados['procesado']
            data2['config']=data_config
            data2['visualizations']=data_visualizaciones
            data2['process']=data_procesado
            data['cell classifier']=data2

        #Creacion del archivo JSON en la ruta de destino
        dir = ruta_destino
        if(ntpath.basename(ruta_destino)):
            archivo='/' + ntpath.basename(ruta_destino)
            dir=dir.replace(archivo,'')
            nombre_archivo=ntpath.basename(ruta_destino)
        if '.json' in nombre_archivo:
            file_name = nombre_archivo
        else:
            file_name= nombre_archivo +'.json'
        with open(os.path.join(dir, file_name), 'w') as file:
            json.dump(data, file, indent=8)

