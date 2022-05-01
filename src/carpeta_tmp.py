import subprocess
import tempfile
from datetime import date
from datetime import datetime

class carpeta_tmp:

    def crear_carpeta_tmp(self):
        """
        Método que se encarga de crear una carpeta en el directorio /tmp con el nombre del tiempo en milisegundos
        en que ha sido ejecutada.
        Returns: Array
        Devuelve el path donde se localiza la carpeta
        """

        #Establezco la fecha de ejecucion en milisegundos
        fecha=datetime.now().time()
        fecha_act= format(fecha.microsecond)
        #Creo la carpeta temporal con el nombre asignaddo
        td = tempfile.mkdtemp(
            suffix="",
            prefix=fecha_act + "_")

        return td

    def borrar_carpeta_tmp(ruta_carpeta_tmp):
        """
        Método encargado de borrar la carpeta temporal
        Returns:0
        """
        try:

            #Ejecuto el comando para borrar la carpeta temporal que se ha creado previamente
            cmd = 'rm -r' + ' ' + ruta_carpeta_tmp
            print("cmd: %s" % cmd)
            proc = subprocess.Popen(cmd.encode('utf-8'), shell=True, stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = proc.communicate()

            return 0

        except:
            print("Error borrando la carpeta tmp")