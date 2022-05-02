import unittest

import os
from src.depurar_datos_source import Depurar_datos
from src.obtener_metadatos_code import Obtener_metadatos_code
from src.cargar_datos_code import Cargar_datos_code
from src.cargar_datos_markdown import Cargar_datos_markdown
from src.cargar_datos_metadata import Cargar_datos_metadata
from src.carpeta_tmp import *
from src.extraer_paths import Extraer_paths


class test(unittest.TestCase):
    def test_cargar_celda_code(self):
        output=[{'cell_type': 'code', 'execution_count': 1, 'id': 'dae545c6', 'metadata': {}, 'outputs': [],
                 'source': ['x=0\n', 'x=x+1\n', 'y=2']}, {'cell_type': 'code', 'execution_count': 2, 'id': '70a75e2e',
                                                          'metadata': {}, 'outputs': [], 'source': ['import numpy']}]
        ruta = os.path.abspath(os.path.dirname(__file__))
        diccionario_actual_out=Cargar_datos_code.cargar_jupyter_code(ruta+"/notebook_test/test_carga_datos.ipynb")
        for i,_ in enumerate(output):
            self.assertDictEqual(output[i],diccionario_actual_out[i])

    def test_cargar_celda_code_connum(self):
        output= {'celdas': [['x=0\n', 'x=x+1\n', 'y=2'], ['import numpy']], 'num_celdas': [1, 2]}

        ruta = os.path.abspath(os.path.dirname(__file__))
        diccionario_actual_out = Cargar_datos_code.cargar_jupyter_code_connum(ruta + "/notebook_test/test_carga_datos.ipynb")
        self.assertDictEqual(output,diccionario_actual_out)

    def test_cargar_celda_markdown(self):
        output= {'cell_type': 'markdown', 'id': '7315a4e1', 'metadata': {}, 'source': ['Ejemplo de celda de texto']}
        ruta=os.path.abspath(os.path.dirname(__file__))
        diccionario_actual_out = Cargar_datos_markdown.cargar_jupyter_markdown(ruta + "/notebook_test/test_carga_datos.ipynb")
        self.assertDictEqual(output,diccionario_actual_out[0])

    def test_cargar_celda_metadata_vacia(self):
        output= [[{'name': 'Daniel'}, {'name': 'Juan'}], 'test notebook']
        ruta = os.path.abspath(os.path.dirname(__file__))
        diccionario_actual_out = Cargar_datos_metadata.cargar_jupyter_metadata( ruta + "/notebook_test/test_carga_datos.ipynb")
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
        cadena_codigo = Cargar_datos_code.cargar_jupyter_code(ruta + "/notebook_test/test_carga_datos.ipynb")
        cadena_source = Obtener_metadatos_code.obtener_source(cadena_codigo)
        cadena_source = Depurar_datos.eliminar_espacios_saltoslinea(cadena_source)
        paths=Extraer_paths.extraer_path(cadena_source)
        assert output==paths




if __name__ == '__main__':
    unittest.main()
