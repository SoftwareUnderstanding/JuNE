
class Extraer_paths:

    def __init__(self,cadena_code):
        self.cadena_code=cadena_code


    def busqueda_paths(cadena_code):
        """
        Método encargado de buscar todos los input paths de un notebook.
        Returns:
        Una lista con los paths obtenidos y otra lista con las variables que contienen paths a buscar
        """

        #Definimos las terminaciones que pueden contener la carga de archivos
        terminaciones_apertura=["open",".reader",".read_csv","read","readline","readlines","from_file","read_pickle"
            ,"requests.get","read_data_sets",".load","CSVLogger",".load_weights",".loadtxt","Image"]
        #Definimos la cadena que contendra los paths que debemos buscar
        cadena_busqueda= []
        #Cadena que contiene los paths extraidos actuales
        cadena_paths = []

        #Recorremos la cadena que contiene las celdas de código
        for i in cadena_code:
            #Recorremos cada linea de cada celda de código
            for x in i:
                #Comprobamos que no sea una linea comentada o un import
                if( not x.startswith('#') and not x.startswith('from') and not x.startswith('import')):
                    posicion_comillas=-1
                    posicion_terminaciones=-1
                    #Buscamos en cada línea si se encuentran las terminaciones previamente definidas
                    for j in terminaciones_apertura:
                        resultado_terminaciones=x.find(j)
                        #Si encuentro las terminaciones
                        if (resultado_terminaciones!=-1):
                            #Compruebo si hay comillas o no la posicion despues del parentesis para conocer si es una variable o el path
                            if(x[resultado_terminaciones+len(j)+1] == '"' or x[resultado_terminaciones+len(j)+1] == "'"):
                                posicion_comillas=resultado_terminaciones+len(j)
                            else:
                                #Si no se ha encontrado un path
                                #Controlamos que contenga una variable
                                if(x[resultado_terminaciones+len(j)+1] != ")"):
                                    posicion_terminaciones=resultado_terminaciones+len(j)



                    #Añadimos a las correspondientes cadenas segun si es un path o variable
                    if(posicion_comillas!=-1):
                        #Añadimos la linea y la posicion donde comienza el path
                        cadena_paths.append([x,posicion_comillas])
                    elif (posicion_terminaciones!=-1):
                        #Añadimos la linea, la celda y la posicion donde comienza la variable
                        cadena_busqueda.append([i,x,posicion_terminaciones])


        return cadena_busqueda,cadena_paths

    def extraccion_paths(cadena_source):
        cadena_busqueda,cadena_paths=Extraer_paths.busqueda_paths(cadena_source)
        cadena_paths_final= []
        paths_variables = []
        #Definicion de las terminaciones de los archivos que puede contener un path
        terminaciones = [".csv", ".pkl", ".pdf", ".txt", ".png", ".aux", ".avi", ".css", ".cvs", ".doc", ".exe", ".gif",
                         ".html", ".jar", ".jpg", ".jpeg", ".mp3", ".mp4", ".raw", ".url", ".zip", ".wav", ".shp",
                         ".geojson", ".xlsx", ".mdd",".npy",".h5",".npz",".log"]

        #Recorremos la cadena que contiene los paths encontrados
        for i in cadena_paths:
            #Extraemos los parametros
            posicion = i[1]
            texto = i[0]
            #Realizamos una busqueda del parentesis final o una coma en caso de que la llamada donde se encuentra contenga mas parámetros.
            posicion_coma=texto.find(",")
            posicion_parentesis=texto.find(")")

            #Si encontramos coma y paréntesis
            if(posicion_coma!=-1 and posicion_parentesis!=-1):
                #Tomamos como referencia la posicion de la coma y extraemos desde donde comienza el path hasta la coma
                texto_path= texto[posicion+1:posicion_coma]
                #Eliminamos las dobles comillas y comillas que pueda haber
                texto_path= texto_path.replace("'", '')
                texto_path= texto_path.replace('"', '')
                #Lo añadimos a la cadena final
                cadena_paths_final.append(texto_path)
            #Si no hay coma y hay parentésis
            if(posicion_coma==-1 and posicion_parentesis!=-1):
                #Extraemos el texto desde la posicion inicial del path hasta el paréntesis
                texto_path = texto[posicion+1:posicion_parentesis]
                #Eliminamos las dobles comillas y comillas que pueda haber
                texto_path = texto_path.replace("'", '')
                texto_path = texto_path.replace('"', '')
                # Lo añadimos a la cadena final
                cadena_paths_final.append(texto_path)

        #Recorremos la cadena_busqueda para poder encontrar el path asociado a la variables que detectamos anteriormente
        for x in cadena_busqueda:
            #Extraemos los datos
            limite=0
            posicion=x[2]
            linea=x[1]
            #Buscamos si contienen parentésis o coma para extraer el nombre de la variable
            posicion_parentesis=linea.find(')')
            posicion_coma=linea.find(',')
            celda=x[0]

            #Si no contiene coma
            if(posicion_coma==-1):
                #Extraemos el nombre de la variable desde el inicio del parentesis al final del paréntesis
                nombre_var=linea[posicion+1:posicion_parentesis]
            else :
                #Extraemos el nombre de la variable desde el principio del paréntesis hasta la coma
                nombre_var = linea[posicion+1:posicion_coma]


            #Una vez obtenida la variable debemos saber si es una llamada del tipo variable=path dentro de la propia llamada
            variable_con_ruta=nombre_var.find('=')
            if(variable_con_ruta!=-1):
                #Si es del tipo mencionado anteriormente
                #Si no hay coma es decir no hay parámetros
                if (posicion_coma==-1):
                    #EL texto se obtiene de la posicion seguido del = hasta el parentesis
                    texto=linea[variable_con_ruta+posicion+2:posicion_parentesis]
                    #Eliminamos las dobles comillas y comillas que pueda haber
                    texto = texto.replace("'", '')
                    texto = texto.replace('"', '')
                    #Añadimos el path a la cadena final
                    paths_variables.append(texto)
                else:
                    #Si contiene coma es decir mas parámetros
                    #Extraemos desde la posicion seguida del = hasta la coma
                    texto=linea[variable_con_ruta+posicion+2:posicion_coma]
                    # Eliminamos las dobles comillas y comillas que pueda haber
                    texto = texto.replace("'", '')
                    texto = texto.replace('"', '')
                    # Añadimos el path a la cadena final
                    paths_variables.append(texto)
            else:
                #Si no cumple con el patron es decir unicamente contiene el nombre de la variable
                #Buscamos la variable dentro de la celda para localizar el path
                for x in celda:
                    #Eliminamos los espacios
                    x_sin_espacios=x.replace(" ",'')
                    #Buscamos la cadena nombre_variable=
                    busqueda_var=x_sin_espacios.find(nombre_var+"=")
                    #Si encontramos la cadena
                    if(busqueda_var!=-1):
                        #Obtenemos la posicion donde comienza el path
                        posicion=busqueda_var+len(nombre_var+"=")
                        #Extraemos el path
                        texto=x[posicion+1:]
                        # Eliminamos las dobles comillas y comillas que pueda haber
                        texto = texto.replace("'",'')
                        texto = texto.replace('"','')
                        #Comprobamos si el path hace referencia a un archivo comprobando las terminaciones de archivos
                        for i in terminaciones:
                            terminacion=texto.find(i)
                            #Comprobamos si es un path de un archivo o una ruta http o https.
                            if(terminacion!=-1 or texto.startswith(' https:') or texto.startswith(' http:')):
                                #Evitamos repeticiones dentro de una misma celda
                                if(limite<1):
                                    paths_variables.append(texto)
                                    limite=limite+1

        input_paths=[]
        for i in paths_variables:
            #Eliminamos los posibles espacios
            i_sin_espacios = i.replace(" ", '')
            #Comprobamos que no haya paths repetidos
            if not i in input_paths:
                input_paths.append(i_sin_espacios)
        for l in cadena_paths_final:
            # Comprobamos que no haya paths repetidos
            if not i in input_paths:
                input_paths.append(l)
        return input_paths


