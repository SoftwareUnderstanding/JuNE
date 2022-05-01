class Obtener_imports_bashcode:
    def __init__(self,source_code):
        self.source_code=source_code

    def obtener_imports(source_code):
        """
        Metodo para obtener los imports de un notebook
        Returns:
        Una lista con los imports que forman el notebook
        """
        imports=[]
        for i in source_code:
            for x in i:
                if x.startswith('import') | x.startswith('from'):
                    imports.append(x)
        return imports

    def obtener_bash_code(source_code):
        """
        Método para extraer el codigo bash de una cadena
        Returns:
        Una lista con las cadenas de bash encontradas
        """
        bash = []
        for i in source_code:
            for x in i:
                if x.startswith('!') or x.startswith('%'):
                    bash.append(x)
        return bash
