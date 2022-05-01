

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
        cadena_paths_sincomillas=[]
        cadena_paths_verificados=[]

        for i in cadena_code:
            for x in i:
                resultado=x.find('"')
                resultado2=x.find("'")
                if(resultado!=-1):
                    cadena_paths.append([x,resultado])
                if(resultado2!=-1):
                    cadena_paths.append([x, resultado2])
        for i in cadena_paths:
            posicion = i[1]
            texto = i[0]
            texto=texto.replace("'",'')
            texto = texto.replace('"', '')

            if(texto.endswith(")") and not texto.startswith("#")):
                cadena_paths_sincomillas.append(texto[posicion:-1])
            elif (texto.startswith("#")):
                pass
            else:
                cadena_paths_sincomillas.append(texto[posicion:])
        for i in cadena_paths_sincomillas:
            terminaciones=[".csv",".pkl",".pdf",".txt", ".png", ".aux",".avi",".css",".cvs",".doc",".exe",".gif",".html",".jar",".jpg"
                           ,".jpeg",".mp3",".mp4",".raw",".url",".zip",".wav",".shp",".geojson",".xlsx",".mdd"]
            terminacion=0
            for x in terminaciones:
                if(i.find(x)!=-1):
                        terminacion+=1

            if (terminacion>0):
                separacioncoma=i.find(',')
                separacionformat=i.find('.format')
                separacionas=i.find('as')
                separacionas2= i.find(') as')
                if(separacioncoma==-1 and separacionformat==-1 and separacionas==-1):
                    cadena_paths_verificados.append(i)
                else:
                    if (separacioncoma!=-1):
                            cadena_paths_verificados.append(i[:separacioncoma])
                    elif(separacionformat!=-1):
                            cadena_paths_verificados.append((i[:separacionformat]))
                    elif (separacionas!=-1):
                            #Elimino desde el espacio hacia posterior
                            cadena_paths_verificados.append(i[:separacionas-2])
                    else:
                            cadena_paths_verificados.append(i[:separacionas2])
        return cadena_paths_verificados
