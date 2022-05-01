class Obtener_metadatos_markdown:

    def __init__(self,cadena_markdown):
        self.cadena_markdown=cadena_markdown

    def obtener_attachments(cadena_markdown):
        """
        Método para obtener los attachments de un notebook
        Returns:
        Una lista con los attachments encontrados
        """
        metadatos_attachments=[]
        for i in cadena_markdown:
         try:
             metadatos_attachments.append(i['attachments'])
         except:
            pass

        return metadatos_attachments

    def obtener_source(cadena_markdown):
        """
        Método para obtener el texto de una celda de tipo markdown
        Returns:
        Una lista con las cadenas extraidas.
        """
        metadatos_source=[]
        metadatos_source.append(cadena_markdown[0]['source'])

        return metadatos_source


    def obtener_metadata(cadena_markdown):
        """
        Método para obtener los metadatos de un notebook
        Returns:
        Una lista con los metadatos extraidos
        """
        metadatos_metadata=[]
        for i in cadena_markdown:
            metadatos_metadata.append(i['metadata'])

        return metadatos_metadata
