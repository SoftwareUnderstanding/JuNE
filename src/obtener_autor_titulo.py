class Obtener_autor_titulo:

    def __init__(self,cadena_metadata):
        self.cadena_metadata=cadena_metadata

    def obtener_autor(cadena_metadata):
        """
        Método encargado de extraer el autor de un notebook
        Returns:
        EL autor del notebook o cadena vacia en caso de no encontrarlo
        """
        autores=cadena_metadata[0]
        lista_autores=[]
        i = 0
        while i < len(autores):
            lista_autores.append(autores[i]['name'])
            i = i + 1
        return lista_autores
    def obtener_titulo(cadena_metadata):
        """
        Método encargado de obtener el titulo de un notebook
        Returns:
        El titulo del notebook o cadena vacia en caso de no encontrarlo
        """
        return cadena_metadata[1]
