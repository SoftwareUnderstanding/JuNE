import unittest

import os
from src.cargar_datos_code import Cargar_datos_code

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
        return 0

if __name__ == '__main__':
    unittest.main()
