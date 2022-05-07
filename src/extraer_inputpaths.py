
class Extraer_paths:

    def __init__(self,cadena_code):
        self.cadena_code=cadena_code


    def extraer_path(cadena_code):
        """
        Método encargado de extraer todos los input paths de un notebook
        Returns:
        Una lista con los paths obtenidos
        """
        cadena_paths=[]
        terminaciones_apertura=["open",".reader",".read_csv","read","readline","readlines"]
        cadena_busqueda= []


        for i in cadena_code:
            for x in i:
                posicion_comillas=-1
                posicion_terminaciones=-1
                for j in terminaciones_apertura:
                    resultado_terminaciones=x.find(j)
                    #Encuentro open o lo q sea
                    if (resultado_terminaciones!=-1):
                        #Compruebo si hay comillas o no la posicion despues del parentesis
                            if(x[resultado_terminaciones+len(j)+1] == '"' or x[resultado_terminaciones+len(j)+1] == "'"):
                                posicion_comillas=resultado_terminaciones+len(j)
                            else:
                                #Controlamos que contenga variables o paths la llamada antes posibles llamadas file.read().
                                if(x[resultado_terminaciones+len(j)+1] != ")"):
                                    posicion_terminaciones=resultado_terminaciones+len(j)

                if(posicion_comillas!=-1):
                    cadena_paths.append([x,posicion_comillas])
                elif (posicion_terminaciones!=-1):
                    cadena_busqueda.append([i,x,posicion_terminaciones])

        return cadena_busqueda,cadena_paths

    def transformaciones_cadenas(cadena_source):
        cadena_busqueda,cadena_paths=Extraer_paths.extraer_path(cadena_source)
        cadena_paths_final= []
        paths_variables = []
        terminaciones = [".csv", ".pkl", ".pdf", ".txt", ".png", ".aux", ".avi", ".css", ".cvs", ".doc", ".exe", ".gif",
                         ".html", ".jar", ".jpg"
            , ".jpeg", ".mp3", ".mp4", ".raw", ".url", ".zip", ".wav", ".shp", ".geojson", ".xlsx", ".mdd"]

        for i in cadena_paths:
            posicion = i[1]
            texto = i[0]
            posicion_coma=texto.find(",")
            posicion_parentesis=texto.find(")")

            #Es decir hay coma y hay parentesis por ejemplo open("ruta,'r')
            if(posicion_coma!=-1 and posicion_parentesis!=-1):
                texto_path= texto[posicion+1:posicion_coma]
                texto_path= texto_path.replace("'", '')
                texto_path= texto_path.replace('"', '')
                cadena_paths_final.append(texto_path)
            if(posicion_coma==-1 and posicion_parentesis!=-1):
                texto_path = texto[posicion+1:posicion_parentesis]
                texto_path = texto_path.replace("'", '')
                texto_path = texto_path.replace('"', '')
                cadena_paths_final.append(texto_path)

        for x in cadena_busqueda:
            posicion=x[2]
            linea=x[1]
            posicion_parentesis=linea.find(')')
            posicion_coma=linea.find(',')
            celda=x[0]
            if(posicion_coma==-1):
                nombre_var=linea[posicion+1:posicion_parentesis]
            else:
                nombre_var = linea[posicion+1:posicion_coma]


            for x in celda:
                x_sin_espacios=x.replace(" ","")
                busqueda_var=x_sin_espacios.find(nombre_var+"=")
                if(busqueda_var!=-1):
                    posicion=busqueda_var+len(nombre_var+"=")
                    texto=x[posicion+1:]
                    texto = texto.replace("'", '')
                    texto = texto.replace('"', '')
                    for i in terminaciones:
                        terminacion=texto.find(i)
                        if(terminacion!=-1):
                            paths_variables.append(texto)

        input_paths=[]
        for i in paths_variables:
            input_paths.append(i)
        for l in cadena_paths_final:
            input_paths.append(l)
        return input_paths


