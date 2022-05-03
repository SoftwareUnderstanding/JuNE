import json
import sys
from os import path
import os
import subprocess

class Futurize:
    def __init__(self):
        self
    def convertir_python3(ruta):
        """
        Método encargado de ejecutar el comando para llamar a la herramienta Futurize para convertir
        de Python2 a Python3
        Returns:0
        """
        try:
            #Ejecuto el comando para utilizar futurize
            cmd = 'futurize --stage2' + ' ' + ruta
            print("cmd: %s" % cmd)
            proc = subprocess.Popen(cmd.encode('utf-8'), shell=True, stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = proc.communicate()

            return 0

        except:
            print("Error en futurize")
            return -1