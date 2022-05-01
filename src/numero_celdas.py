

class Numero_celdas:

    """constructor"""
    def __init__(self,cadena_code, cadena_markdown):
        self.cadena_code=cadena_code
        self.cadena_markdown=cadena_markdown


    def numero_celdas_code_bash(cadena_code,cadena_markdwon):
        """
        Método encargado de contabilizar el numero de lineas de codigo , markdown y bash
        Args:
            cadena_markdwon: Cadena de texto

        Returns:
        Devuelve un diccionario con el numero de líneas de tipo codigo , markdown y bash
        """
        tipo_lineas ={}
        lineas_texto=0
        celda_bash=0
        celda_codigo=0
        lineas_bash = 0
        lineas_python = 0

        for i in cadena_code:
            if (lineas_bash > 0 & lineas_python > 0):
                celda_bash += 1
                celda_codigo += 1
            else:
                if (lineas_bash > 0):
                    celda_bash += 1
                else:
                    if (lineas_python > 0):
                        celda_codigo += 1

            lineas_bash = 0
            lineas_python = 0
            for x in i['source']:
                if x.startswith('!'):
                    lineas_bash+=1
                else:
                    lineas_python+=1


        for i in cadena_markdwon:
            if(i['cell_type']=='markdown'):
                lineas_texto=lineas_texto+1

        if (lineas_bash > 0 & lineas_python > 0):
            celda_bash += 1
            celda_codigo += 1
        else:
            if (lineas_bash > 0):
                celda_bash += 1
            else:
                if (lineas_python > 0):
                    celda_codigo += 1
                    
        tipo_lineas['codigo']=(celda_codigo)
        tipo_lineas['texto']=(lineas_texto)
        tipo_lineas['bash']=(celda_bash)

        return tipo_lineas



