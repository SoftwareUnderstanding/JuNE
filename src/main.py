import ntpath
import os
import click

from cargar_datos_code import Cargar_datos_code as Cargar_datos_code
from cargar_datos_markdown import Cargar_datos_markdown as Cargar_datos_markdown
from cargar_datos_metadata import Cargar_datos_metadata as Cargar_datos_metadata
from depurar_datos_source import Depurar_datos as Depurar_datos
from escritura_JSON import Escritura_JSON as Escritura_JSON
from inspect4py import Inspect4py as Inspect4py
from jupyteraPython import JupyteraPython as JupyteraPython
from lectura_JSON import Lectura_JSON as Lectura_JSON
from obtener_autor_titulo import Obtener_autor_titulo as Obtener_autor_titulo
from obtener_imports_bashcode import Obtener_imports_bashcode as Obtener_imports_bashcode
from obtener_metadatos_code import Obtener_metadatos_code as Obtener_metadatos_code
from futurize import Futurize as Futurize
from carpeta_tmp import carpeta_tmp as carpeta_tmp
from codebert_train import codebert_train as codebert_train
from binario_clasificador import binario_clasificador
from clasificadores import Clasificadores as Clasificadores
from extraer_inputpaths import Extraer_paths as Extraer_paths

def crear_output_dir(output_dir, input_path):
    control=os.path.abspath(os.getcwd())
    dir=output_dir
    archivo = '/' + ntpath.basename(output_dir)
    dir = dir.replace(archivo, '')
    if output_dir == 'output_dir':
        output_dir=control+'/'
    else:
        if not os.path.exists(dir):
            print("El directorio seleccionado no existe")
            output_dir=-1
        else:
            if not ".json" in output_dir:
                print("Ha introducido el path incorrectamente , debe llevar el formato /***/... /nombre_archivo.json")
                output_dir=-1

    return output_dir

@click.command()
@click.option('-i', '--input_path', type=str, required=True, help = "Direccion de entrada del notebook a inspeccionar")
@click.option('-tmp', '--tmp_dir', type=bool, is_flag=True, help="Opcion para eliminar la carpeta temporal")
@click.option('-o', '--output_dir', type=str, default='output_dir', help="Direccion de salida de los metadatos extraidos")
@click.option('-inspect', '--inspect', is_flag=True, help='Ejecución mediante inspect4py')

def main(input_path, tmp_dir, output_dir,inspect):

    #Compruebo si el directorio de entrada existe
    #COmentario prueba
    #COMENTARIO PRUEBA2
    if os.path.isfile(input_path):
        #Compruebo si el archivo no esta vacio
        if os.stat(input_path).st_size!=0:
            output_dir=crear_output_dir(output_dir, input_path)
            #Compruebo si existe ruta de destino y es correcta
            if(output_dir!=-1):
                # Creacion carpeta tmp
                carpeta_temporal = carpeta_tmp()
                ruta_carpeta_tmp = carpeta_temporal.crear_carpeta_tmp()
                nombre_carpeta = ntpath.basename(ruta_carpeta_tmp)

                # Establezco las rutas que voy a usar
                nombre_archivo = ntpath.basename(input_path)
                posicion = nombre_archivo.find('.ipynb')
                nombre_archivo = nombre_archivo[:posicion]
                ruta_carpeta_python = ruta_carpeta_tmp + "/" + nombre_archivo + ".py"
                ruta_JSON = ruta_carpeta_tmp + "/directory_info.json"
                ruta_JSON2 = ruta_carpeta_tmp + "/" + nombre_carpeta + "/json_files/" + nombre_archivo + ".json"
                #Compruebo si se ha seleccionado la opcion inspect
                if(inspect):
                    # Convierto el Jupyter a Python
                    JupyteraPython.convert(input_path,ruta_carpeta_python)
                    # Convierto el archivo a Python3 para poder usar Inspect4py
                    Futurize.convertir_python3(ruta_carpeta_python)
                    # Extraigo mediante inspect4py los metadatos del archivo .py
                    Inspect4py.extract_requirements(ruta_carpeta_tmp,ruta_carpeta_tmp)
                    # Extraigo del JSON generado por inspect4py las depencias y la descripcion
                    requerimientos = Lectura_JSON.obtener_dependencias(ruta_JSON)
                    # Extraigo la descripcion
                    descripcion = Lectura_JSON.obtener_descripcion(ruta_JSON2)
                    # Si inspect4py no ha sido capaz de extraer la descripcion utilizo la primera línea de texto
                    if len(descripcion)==0:
                            cadena_markdown = Cargar_datos_markdown.cargar_jupyter_markdown(input_path)
                            if cadena_markdown:
                                descripcion=cadena_markdown[0]['source']
                                descripcion = Depurar_datos.depurar_descripcion(descripcion)
                            else:
                                print("No dispone de texto para extraer la descripción")
                                descripcion=[]

                    # Extraigo las llamadas a funciones que se realizan
                    llamadas = Lectura_JSON.obtener_llamadas_funciones(ruta_JSON2)
                    # Cargo el archivo y extraigo sus respectivas celdas de codigo y texto
                    cadena_codigo = Cargar_datos_code.cargar_jupyter_code(input_path)
                    cadena_metadata = Cargar_datos_metadata.cargar_jupyter_metadata(input_path)
                    cadena_source = Obtener_metadatos_code.obtener_source(cadena_codigo)
                    cadena_source = Depurar_datos.eliminar_espacios_saltoslinea(cadena_source)
                    #Extraigo los input paths que haya en el archivo
                    cadena_paths=Extraer_paths.transformaciones_cadenas(cadena_source)

                    # Obtengo el titulo y el autor del Notebook
                    if len(cadena_metadata) != 0:
                        autor = Obtener_autor_titulo.obtener_autor(cadena_metadata)
                        titulo = Obtener_autor_titulo.obtener_titulo(cadena_metadata)
                    else:
                        autor = []
                        titulo = []

                    # Obtengo las lineas de codigo bash depurando antes las lineas para eliminar saltos de linea
                    bash = Obtener_imports_bashcode.obtener_bash_code(cadena_source)
                    #Llamo al clasificador para clasificar las celdas.
                    ruta_modelos= os.path.abspath(os.path.dirname(__file__))
                    visualizaciones=0;
                    imports=0;

                    #Bag of words
                    # Clasificadores.tokenizar(ruta_modelos)
                    #visualizaciones,imports=clasificadores.clasificacion(ruta_modelos,cadena_source)


                    #CLASIFICADOR MULTIVARIANTE
                    #codebert_train.entrenamiento(ruta_modelos)

                    #CLASIFICADORES BINARIO
                    #Esta es la llamada que se realiza para la obtencion del entrenamiento de los clasificadores
                    entities_embed_configuracion, entity_classes_configuracion, entities_embed_visualizacion, \
                    entity_classes_visualizacion, entities_embed_procesado, entity_classes_procesado= binario_clasificador.embeddings(ruta_modelos)

                    binario_clasificador.entrenamiento(ruta_modelos,entities_embed_configuracion, entity_classes_configuracion, entities_embed_visualizacion, \
                    entity_classes_visualizacion, entities_embed_procesado, entity_classes_procesado)

                    #Cargo las celdas del notebook a analizar y sus numeros de celda
                    cadena_analizar=Cargar_datos_code.cargar_jupyter_code_connum(input_path)
                    #Llamo al metodo correspondiente para clasificar las celdas
                    resultados=binario_clasificador.clasificacion(ruta_modelos,cadena_analizar)
                    #resultados=codebert_train.clasificacion(ruta_modelos,cadena_analizar)

                    # Escribo el JSON con toda la informacion relevante del notebook
                    Escritura_JSON.escribir_JSON(output_dir,bash, autor, titulo, requerimientos, descripcion,llamadas,
                                                 cadena_paths,nombre_archivo,1,imports,resultados)
                else:
                    #Crago las celdas de texto del notebook
                    cadena_markdown = Cargar_datos_markdown.cargar_jupyter_markdown(input_path)
                    if cadena_markdown:
                        #Obtengo la descripcion
                        descripcion = cadena_markdown[0]['source']
                        #Elimino saltos de linea
                        descripcion_depurada= Depurar_datos.depurar_descripcion(descripcion)
                    else:
                        print("No dispone de texto para extraer la descripción")
                        descripcion_depurada=[]
                    #Cargo los metadatos para comprobar si hay autores del notebook
                    cadena_metadata = Cargar_datos_metadata.cargar_jupyter_metadata(input_path)
                    if len(cadena_metadata) != 0:
                        autor = Obtener_autor_titulo.obtener_autor(cadena_metadata)
                        titulo = Obtener_autor_titulo.obtener_titulo(cadena_metadata)
                    else:
                        autor = []
                        titulo = []

                    #Cargo las celdas de código y las depuro eliminado saltos y espacios innecesarios.
                    cadena_codigo = Cargar_datos_code.cargar_jupyter_code(input_path)
                    cadena_source = Obtener_metadatos_code.obtener_source(cadena_codigo)
                    cadena_source = Depurar_datos.eliminar_espacios_saltoslinea(cadena_source)
                    # Obtengo las lineas de codigo bash depurando antes las lineas para eliminar saltos de linea
                    bash = Obtener_imports_bashcode.obtener_bash_code(cadena_source)
                    cadena_paths = Extraer_paths.transformaciones_cadenas(cadena_source)

                    # Llamo al clasificador para clasificar las celdas.
                    ruta_modelos = os.path.abspath(os.path.dirname(__file__))
                    visualizaciones=0
                    imports=0
                    # Bag of words
                    # Clasificadores.tokenizar(ruta_modelos)
                    # visualizaciones,imports=clasificadores.clasificacion(ruta_modelos,cadena_source)

                    # CLASIFICADOR MULTIVARIANTE
                    # codebert_train.entrenamiento(ruta_modelos)

                    # CLASIFICADOR BINARIO
                    #Esta es la llamada que se realiza para la obtencion del entrenamiento de los clasificadores
                    entities_embed_configuracion, entity_classes_configuracion, entities_embed_visualizacion, \
                    entity_classes_visualizacion, entities_embed_procesado, entity_classes_procesado = binario_clasificador.embeddings(
                        ruta_modelos)

                    binario_clasificador.entrenamiento(ruta_modelos, entities_embed_configuracion,
                                                       entity_classes_configuracion, entities_embed_visualizacion, \
                                                       entity_classes_visualizacion, entities_embed_procesado,
                                                      entity_classes_procesado)

                    # Cargo las celdas del notebook a analizar y sus numeros de celda
                    cadena_analizar = Cargar_datos_code.cargar_jupyter_code_connum(input_path)
                    # Llamo al metodo correspondiente para clasificar las celdas
                    resultados = binario_clasificador.clasificacion(ruta_modelos, cadena_analizar)
                    # resultados=codebert_train.clasificacion(ruta_modelos,cadena_analizar)

                    # Escribo el JSON con toda la informacion relevante del notebook
                    Escritura_JSON.escribir_JSON(output_dir, bash, autor, titulo, None, descripcion_depurada, None,
                                                 cadena_paths, nombre_archivo, 1, imports,resultados)

                #Compruebo si se ha ejecutado la opcion tmp_dir
                if(tmp_dir):
                    #Elimino la carpeta temporal
                    carpeta_tmp.borrar_carpeta_tmp(ruta_carpeta_tmp)
        else:
            print("El archivo especificado está vacio")
    else:
        print("El archivo especificado no existe")


if __name__ == "__main__":
   main()