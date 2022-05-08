import unittest
import os
from src.extraer_inputpaths import Extraer_paths
from src.depurar_datos_source import Depurar_datos
from src.obtener_metadatos_code import Obtener_metadatos_code
from src.cargar_datos_code import Cargar_datos_code

class test_input_paths(unittest.TestCase):
    def test_1(self):
        output = ['R315114383A.mdd', 'prueba1.csv', 'prueba2.csv', 'prueba3.txt']
        ruta = os.path.abspath(os.path.dirname(__file__))
        cadena_codigo = Cargar_datos_code.cargar_jupyter_code(ruta + "/test_data/test_path1.ipynb")
        cadena_source = Obtener_metadatos_code.obtener_source(cadena_codigo)
        cadena_source = Depurar_datos.eliminar_espacios_saltoslinea(cadena_source)
        paths = Extraer_paths.extraccion_paths(cadena_source)
        assert output == paths

    def test_2(self):
        output =['https://maps.googleapis.com/maps/api/directions/json?origin=%s,%s&%skey=%s%(lat,lon,argstr,my_apikey)',
                 'data/ZIP_CODE_040114/ZIP_CODE_040114.shp', 'data-outputs/rides_summary.pkl', 'durations_by_zip1.geojson']

        ruta = os.path.abspath(os.path.dirname(__file__))
        cadena_codigo = Cargar_datos_code.cargar_jupyter_code(ruta + "/test_data/test_path2.ipynb")
        cadena_source = Obtener_metadatos_code.obtener_source(cadena_codigo)
        cadena_source = Depurar_datos.eliminar_espacios_saltoslinea(cadena_source)
        paths = Extraer_paths.extraccion_paths(cadena_source)
        assert output == paths

    def test_3(self):
        output =["MNIST_data/"]
        ruta = os.path.abspath(os.path.dirname(__file__))
        cadena_codigo = Cargar_datos_code.cargar_jupyter_code(ruta + "/test_data/test_path3.ipynb")
        cadena_source = Obtener_metadatos_code.obtener_source(cadena_codigo)
        cadena_source = Depurar_datos.eliminar_espacios_saltoslinea(cadena_source)
        paths = Extraer_paths.extraccion_paths(cadena_source)
        assert output == paths

    def test_4(self):
        output=['newdataset1.npz', 'xtesting.npy', 'ytesting.npy', 'weight_newarch_epoch_300_coarse.h5', 'testingnewarch.log']
        ruta = os.path.abspath(os.path.dirname(__file__))
        cadena_codigo = Cargar_datos_code.cargar_jupyter_code(ruta + "/test_data/test_path4.ipynb")
        cadena_source = Obtener_metadatos_code.obtener_source(cadena_codigo)
        cadena_source = Depurar_datos.eliminar_espacios_saltoslinea(cadena_source)
        paths = Extraer_paths.extraccion_paths(cadena_source)
        assert output == paths

    def test_5(self):
        output=["https://localhost:5000/ows/wps?version=1.0.0&service=wps&request=describeprocess&identifier=hello",
        "https://localhost:5000/ows/wps?version=1.0.0&service=wps&request=execute&identifier=hello",
        "https://localhost:5000/ows/proxy/hummingbird?version=1.0.0&service=wps&request=execute&identifier=ncdump"]
        ruta = os.path.abspath(os.path.dirname(__file__))
        cadena_codigo = Cargar_datos_code.cargar_jupyter_code(ruta +"/test_data/test_path5.ipynb")
        cadena_source = Obtener_metadatos_code.obtener_source(cadena_codigo)
        cadena_source = Depurar_datos.eliminar_espacios_saltoslinea(cadena_source)
        paths = Extraer_paths.extraccion_paths(cadena_source)
        assert output == paths

    def test_6(self):
        output=['data/iris_data_clean.csv']
        ruta = os.path.abspath(os.path.dirname(__file__))
        cadena_codigo = Cargar_datos_code.cargar_jupyter_code(ruta + "/test_data/test_path6.ipynb")
        cadena_source = Obtener_metadatos_code.obtener_source(cadena_codigo)
        cadena_source = Depurar_datos.eliminar_espacios_saltoslinea(cadena_source)
        paths = Extraer_paths.extraccion_paths(cadena_source)
        assert output == paths

    def test_7(self):
        output=['inflammation-01.csv']
        ruta = os.path.abspath(os.path.dirname(__file__))
        cadena_codigo = Cargar_datos_code.cargar_jupyter_code(ruta + "/test_data/test_path7.ipynb")
        cadena_source = Obtener_metadatos_code.obtener_source(cadena_codigo)
        cadena_source = Depurar_datos.eliminar_espacios_saltoslinea(cadena_source)
        paths = Extraer_paths.extraccion_paths(cadena_source)
        assert output == paths

    def test_8(self):
        output = ['img/1.png', 'img/2.png', 'img/3.png', 'img/4.png', 'img/5.png', 'img/6.png', 'img/7.png',
                  'img/8.png', 'img/9.png', 'img/10.png', 'img/11.png', 'img/12.png', 'img/13.png', 'img/14.png',
                  'img/15.png', 'img/16.png', 'img/17.png', 'img/18.png', 'img/19.png', 'img/20.png', 'img/21.png',
                  'img/22.png', 'img/23.png', 'img/24.png', 'img/25.png', 'img/26.png', 'img/27.png', 'img/28.png',
                  'img/29.png', 'img/30.png', 'img/31.png', 'img/32.png', 'img/33.png', 'img/34.png', 'img/35.png',
                  'img/36.png', 'img/37.png', 'img/38.png', 'img/39.png', 'img/40.png', 'img/41.png', 'img/42.png',
                  'img/44.png']
        ruta = os.path.abspath(os.path.dirname(__file__))
        cadena_codigo = Cargar_datos_code.cargar_jupyter_code(ruta + "/test_data/test_path8.ipynb")
        cadena_source = Obtener_metadatos_code.obtener_source(cadena_codigo)
        cadena_source = Depurar_datos.eliminar_espacios_saltoslinea(cadena_source)
        paths = Extraer_paths.extraccion_paths(cadena_source)
        assert output == paths
