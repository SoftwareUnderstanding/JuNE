import os
import subprocess


class Inspect4py:

    def __init__(self,input_path,output_path):
        self.input_path=input_path
        self.output_path=output_path

    def extract_requirements(input_path,output_path):
        """
        Método encargado de ejecutar el comando para llamar a la herramienta Inspect4py
        Args:
            output_path: Ruta donde se va a crear los archivos obtenidos tras la ejecución

        Returns:0
        """
        print("Finding the requirements with the inspect4py package for %s" % input_path)
        try:
            file_name = os.path.basename(input_path)

            #Ejecuto el comando para utilizar inspect4py
            cmd = 'inspect4py -r -si -i' + ' ' + input_path +' '+ '-o'+ ' '+ output_path
            # cmd = 'echo n | pigar -P ' + input_path + ' --without-referenced-comments -p ' + file_name
            print("cmd: %s" %cmd)
            proc = subprocess.Popen(cmd.encode('utf-8'), shell=True, stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = proc.communicate()
            return 0

        except:
            print("Error finding the requirements in" % input_path)