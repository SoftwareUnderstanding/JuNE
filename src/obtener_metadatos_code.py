class Obtener_metadatos_code:

    def __init__(self,cadena_code):
        self.cadena_code=cadena_code

    def obtener_executioncount(cadena_code):
        """

        Returns:

        """
        metadatos_executioncount=[]

        #Bucle para recorrer el apartado execution_count de las celdas
        for i in cadena_code:
            metadatos_executioncount.append(i['execution_count'])

        return metadatos_executioncount

    def obtener_outputs(cadena_code):
        metadatos_outputs=[]

        #Bucle para recorrer el apartado outputs de las celdas
        for i in cadena_code:
            metadatos_outputs.append(i['outputs'])
            return metadatos_outputs


    def obtener_metadata(cadena_code):
        metadatos_metadata=[]

        #Bucle para recorrer el apartado metadata de las celdas
        for i in cadena_code:
            metadatos_metadata.append(i['metadata'])

        return metadatos_metadata

    def obtener_source(cadena_code):
        metadatos_source=[]
        for i in cadena_code:
            metadatos_source.append(i['source'])

        return metadatos_source

