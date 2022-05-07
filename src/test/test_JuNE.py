import unittest

import os
from src.depurar_datos_source import Depurar_datos
from src.obtener_metadatos_code import Obtener_metadatos_code
from src.cargar_datos_code import Cargar_datos_code
from src.cargar_datos_markdown import Cargar_datos_markdown
from src.cargar_datos_metadata import Cargar_datos_metadata
from src.carpeta_tmp import *
from src.extraer_inputpaths import Extraer_paths
from src.jupyteraPython import JupyteraPython
from src.futurize import Futurize
from src.inspect4py import Inspect4py
from src.lectura_JSON import Lectura_JSON
from src.obtener_imports_bashcode import *


class test(unittest.TestCase):
    def test_cargar_celda_code(self):
        output=[{'cell_type': 'code', 'execution_count': 1, 'id': 'dae545c6', 'metadata': {}, 'outputs': [], 'source':
            ['x=0\n', 'x=x+1\n', 'y=2']}, {'cell_type': 'code', 'execution_count': 2, 'id': '70a75e2e', 'metadata': {}, 'outputs': [],
             'source': ['import numpy']}, {'cell_type': 'code',
             'execution_count': None, 'id': 'f874254c', 'metadata': {},
             'outputs': [], 'source': ['import csv\n', ' \n', "with open('example.csv', newline='') as File:  \n",
            '    reader = csv.reader(File)\n', '    for row in reader:\n', '        print(row)']}]

        ruta = os.path.abspath(os.path.dirname(__file__))
        diccionario_actual_out=Cargar_datos_code.cargar_jupyter_code(ruta+"/test_data/test_carga_datos.ipynb")
        for i,_ in enumerate(output):
            self.assertDictEqual(output[i],diccionario_actual_out[i])

    def test_cargar_celda_code_connum(self):
        output= {'celdas': [['x=0\n', 'x=x+1\n', 'y=2'], ['import numpy'], ['import csv\n', ' \n',
                "with open('example.csv', newline='') as File:  \n", '    reader = csv.reader(File)\n',
              '    for row in reader:\n', '        print(row)']], 'num_celdas': [1, 2, 4]}


        ruta = os.path.abspath(os.path.dirname(__file__))
        diccionario_actual_out = Cargar_datos_code.cargar_jupyter_code_connum(ruta + "/test_data/test_carga_datos.ipynb")
        self.assertDictEqual(output,diccionario_actual_out)

    def test_cargar_celda_markdown(self):
        output= {'cell_type': 'markdown', 'id': '7315a4e1', 'metadata': {}, 'source': ['Ejemplo de celda de texto']}
        ruta=os.path.abspath(os.path.dirname(__file__))
        diccionario_actual_out = Cargar_datos_markdown.cargar_jupyter_markdown(ruta + "/test_data/test_carga_datos.ipynb")
        self.assertDictEqual(output,diccionario_actual_out[0])

    def test_cargar_celda_metadata_vacia(self):
        output= [[{'name': 'Daniel'}, {'name': 'Juan'}], 'test notebook']
        ruta = os.path.abspath(os.path.dirname(__file__))
        diccionario_actual_out = Cargar_datos_metadata.cargar_jupyter_metadata( ruta + "/test_data/test_carga_datos.ipynb")
        assert output == diccionario_actual_out

    def test_crear_carpeta_tmp(self):
        ruta=carpeta_tmp.crear_carpeta_tmp(self)
        assert os.path.exists(ruta)

    def test_borrar_carpeta_tmp(self):
        ruta=carpeta_tmp.crear_carpeta_tmp(self)
        carpeta_tmp.borrar_carpeta_tmp(ruta)
        assert os.path.exists(ruta)==False

    def test_extraer_paths(self):
        output= ['example.csv']
        ruta = os.path.abspath(os.path.dirname(__file__))
        cadena_codigo = Cargar_datos_code.cargar_jupyter_code(ruta + "/test_data/test_carga_datos.ipynb")
        cadena_source = Obtener_metadatos_code.obtener_source(cadena_codigo)
        cadena_source = Depurar_datos.eliminar_espacios_saltoslinea(cadena_source)
        paths=Extraer_paths.transformaciones_cadenas(cadena_source)
        assert output==paths

    def test_JupyteraPython(self):
        ruta_input = os.path.abspath(os.path.dirname(__file__))+"/test_data/test_carga_datos.ipynb"
        ruta_output= os.path.abspath(os.path.dirname(__file__))+"/test_data/test_carga_datos.py"
        JupyteraPython.convert(ruta_input,ruta_output)
        assert os.path.exists(ruta_output) == True

    def test_Futurize(self):
        ruta=os.path.abspath(os.path.dirname(__file__))+"/test_data/test_carga_datos.py"
        resultado=Futurize.convertir_python3(ruta)
        assert resultado==0

    def test_Inspect4py(self):
        ruta_input = os.path.abspath(os.path.dirname(__file__)) + "/test_data/test_carga_datos.py"
        ruta_output= os.path.abspath(os.path.dirname(__file__)) + "/Inspect4py_data/"
        Inspect4py.extract_requirements(ruta_input,ruta_output)
        assert os.path.exists(ruta_output+"json_files/test_carga_datos.json")
        # file path
        try:
            os.remove(ruta_output + "/json_files/test_carga_datos.json")
            print("% s removed successfully" % ruta_output + "/json_files/test_carga_datos.json")
        except OSError as error:
            print(error)
            print("File path can not be removed")
        os.rmdir(ruta_output +"/json_files/")
        os.rmdir(ruta_output)

    def test_lectura_JSON_obtener_descripcion(self):
        ruta_input = os.path.abspath(os.path.dirname(__file__)) + "/test_data/Metadatos.ipynb"
        ruta_output_py = os.path.abspath(os.path.dirname(__file__)) + "/test_data/Metadatos.py"
        JupyteraPython.convert(ruta_input, ruta_output_py)
        ruta_output = os.path.abspath(os.path.dirname(__file__)) + "/Inspect4py_data/"
        Inspect4py.extract_requirements(ruta_output_py, ruta_output)
        ruta_JSON= ruta_output+ "/json_files/" + "Metadatos.json"
        descripcion_prueba= Lectura_JSON.obtener_descripcion(ruta_JSON)
        descripcion="Ejemplo de una imagen"
        assert descripcion==descripcion_prueba
        try:
            os.remove(ruta_output + "/json_files/Metadatos.json")
            print("% s removed successfully" % ruta_output + "/json_files/Metadatos.json")
        except OSError as error:
            print(error)
            print("File path can not be removed")
        os.rmdir(ruta_output + "/json_files/")
        os.rmdir(ruta_output)

    def test_lectura_JSON_llamadasfunciones(self):
        llamadas= {'calls': ['pandas.DataFrame', 'print', 'matplotlib.plot', 'matplotlib.show'],
                   'store_vars_calls': {'tabla': 'pd.DataFrame'}}
        ruta_input = os.path.abspath(os.path.dirname(__file__)) + "/test_data/Metadatos.ipynb"
        ruta_output_py = os.path.abspath(os.path.dirname(__file__)) + "/test_data/Metadatos.py"
        JupyteraPython.convert(ruta_input, ruta_output_py)
        ruta_output = os.path.abspath(os.path.dirname(__file__)) + "/Inspect4py_data/"
        Inspect4py.extract_requirements(ruta_output_py, ruta_output)
        ruta_JSON = ruta_output + "/json_files/" + "Metadatos.json"
        llamadas_prueba= Lectura_JSON.obtener_llamadas_funciones(ruta_JSON)
        assert llamadas==llamadas_prueba

        try:
            os.remove(ruta_output + "/json_files/Metadatos.json")
            print("% s removed successfully" % ruta_output + "/json_files/Metadatos.json")
        except OSError as error:
            print(error)
            print("File path can not be removed")
        os.rmdir(ruta_output + "/json_files/")
        os.rmdir(ruta_output)

    def test_obtenerbash(self):
        bash=['!pip install pandas']
        ruta_input=os.path.abspath(os.path.dirname(__file__)) + "/test_data/Metadatos.ipynb"
        cadena_codigo = Cargar_datos_code.cargar_jupyter_code(ruta_input)
        cadena_source = Obtener_metadatos_code.obtener_source(cadena_codigo)
        cadena_source = Depurar_datos.eliminar_espacios_saltoslinea(cadena_source)
        bash_prueba=Obtener_imports_bashcode.obtener_bash_code(cadena_source)
        assert bash==bash_prueba

    def test_obtener_metadatos_code_source(self):
        source= [['!pip install pandas\n', 'import pandas as pd\n', '#creacion de una tabla\n',
                  'tabla=pd.DataFrame(data= [[1,1],[2,4],[3,9]],\n', "                    columns = ['numero', 'cuadrado'])\n",
                  'print(tabla)'], ['import matplotlib.pyplot as plt\n', '\n', 'x= [1,2,3,4,5]\n', 'y=[1,4,9,16,25]\n',
                  '\n', 'plt.plot(x,y)\n', 'plt.show()'], ['import csv\n', '\n',
                  "with open('example.csv', newline='') as File:  \n", '    reader = csv.reader(File)\n',
                  '    for row in reader:\n', '        print(row)']]
        ruta_input = os.path.abspath(os.path.dirname(__file__)) + "/test_data/Metadatos.ipynb"
        cadena_codigo = Cargar_datos_code.cargar_jupyter_code(ruta_input)
        cadena_source = Obtener_metadatos_code.obtener_source(cadena_codigo)
        assert source==cadena_source













if __name__ == '__main__':
    unittest.main()
