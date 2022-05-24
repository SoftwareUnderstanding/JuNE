import pickle
import csv
from transformers import AutoTokenizer, AutoModel, TrainingArguments, AutoModelForPreTraining, AutoModelForMaskedLM
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import torch
import sklearn
from sklearn.metrics import f1_score


class binario_clasificador:

        def embeddings(ruta):
                """
                Método encargado de realizar los embeddings para el posterior entrenamiento del clasificador.
                Se realizan los embeddings pertenecientes a los tres archivos de entrenamiento Configuracion.csv,
                Visualizacion.csv y Procesado.csv.
                :return:
                El conjunto de tuplas de embeddings con los correspondientes labels.
                """
                file_configuracion = csv.reader(open(ruta + "/Entrenamiento/Configuracion.csv"), delimiter=';')
                labels_configuracion = []
                celda_configuracion = []
                next(file_configuracion)
                # Recorremos el csv y guardamos los valores de las celdas y los labels
                for line in file_configuracion:
                        labels_configuracion.append(line[0])
                        celda_configuracion.append(line[1])

                file_visualizacion = csv.reader(open(ruta + "/Entrenamiento/Visualizacion.csv"), delimiter=';')
                labels_visualizacion = []
                celda_visualizacion = []
                next(file_visualizacion)
                # Recorremos el csv y guardamos los valores de las celdas y los labels

                for line in file_visualizacion:
                        labels_visualizacion.append(line[0])
                        celda_visualizacion.append(line[1])

                file_procesado = csv.reader(open(ruta + "/Entrenamiento/Procesado.csv"), delimiter=';')
                labels_procesado = []
                celda_procesado = []
                next(file_procesado)

                # Recorremos el csv y guardamos los valores de las celdas y los labels
                for line in file_procesado:
                        labels_procesado.append(line[0])
                        celda_procesado.append(line[1])

                # Definimos el tokenizer y el modelo que vamos a usar
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
                model = AutoModel.from_pretrained("microsoft/codebert-base")
                model.to(device)

                # Realizamos los embeddings de la cadenas de configuracion
                entities_configuracion = celda_configuracion
                entities_tokenize_configuracion = tokenizer(entities_configuracion, return_tensors="pt",
                                                            padding='max_length', truncation=True,
                                                            max_length=30, add_special_tokens=True)
                entities_embed_configuracion = model(entities_tokenize_configuracion.input_ids)[0].prod(
                        dim=1).detach().numpy()
                # Guardamos los labels correspondientes a los embeddings
                entity_classes_configuracion = labels_configuracion

                # Realizamos los embeddings de las cadenas de visualizacion
                entities_visualizacion = celda_visualizacion
                entities_tokenize_visualizacion = tokenizer(entities_visualizacion, return_tensors="pt",
                                                            padding='max_length', truncation=True,
                                                            max_length=30, add_special_tokens=True)
                entities_embed_visualizacion = model(entities_tokenize_visualizacion.input_ids)[0].prod(
                        dim=1).detach().numpy()
                # Guardamos los labels correspondientes a los embeddings
                entity_classes_visualizacion = labels_visualizacion

                # Realizamos los embeddings de la cadenas de procesado
                entities_procesado = celda_procesado
                entities_tokenize_procesado = tokenizer(entities_procesado, return_tensors="pt",
                                                        padding='max_length', truncation=True,
                                                        max_length=30, add_special_tokens=True)
                entities_embed_procesado = model(entities_tokenize_procesado.input_ids)[0].prod(
                        dim=1).detach().numpy()
                # GUardamos los labels correspondientes a los embeddings
                entity_classes_procesado = labels_procesado

                return entities_embed_configuracion, entity_classes_configuracion, entities_embed_visualizacion, entity_classes_visualizacion, entities_embed_procesado, entity_classes_procesado

        def entrenamiento(ruta, entities_embed_configuracion, entity_classes_configuracion,
                          entities_embed_visualizacion,
                          entity_classes_visualizacion, entities_embed_procesado, entity_classes_procesado):
                """
                Método encargado de realizar el entrenamiento de los algoritmos y su posterior guardado.
                :param entities_embed_configuracion: Tupla que contiene los embeddings de las celdas de Configuracion
                :param entity_classes_configuracion: Labels pertenecientes a las celdas de configuración
                :param entities_embed_visualizacion: Tupla que contiene los embeddings de las celdas de Visualización
                :param entity_classes_visualizacion: Labels pertenecientes a las celdas de Visualización
                :param entities_embed_procesado: Tupla que contiene los embeddings de las celdas de Procesado
                :param entity_classes_procesado: Labels pertenecientes a las celdas de Procesado
                :return:
                """

                # Separamos en entrenamiento y test en un 70/30 los embeddings y labels de configuración
                X_train_config, X_test_config, y_train_config, y_test_config = sklearn.model_selection.train_test_split(
                        entities_embed_configuracion,
                        entity_classes_configuracion,
                        train_size=0.7)

                # Separamos en entrenamiento y test en un 70/30 los embeddings y labels de visualización
                X_train_visualizacion, X_test_visualizacion, y_train_visualizacion, y_test_visualizacion = sklearn.model_selection.train_test_split(
                        entities_embed_visualizacion,
                        entity_classes_visualizacion,
                        train_size=0.7)
                # Separamos en entrenamiento y test en un 70/30 los embeddings y labels de procesado
                X_train_procesado, X_test_procesado, y_train_procesado, y_test_procesado = sklearn.model_selection.train_test_split(
                        entities_embed_procesado,
                        entity_classes_procesado,
                        train_size=0.7)

                # Variable donde vamos a almacenar los resultados de los clasificadores (accuracy ,F1 , Mathew)
                results = []
                # Aplicamos los modelos ( entrenamiento y prediccion)
                # DecisionTreeClassifier
                # Cargamos el clasificador
                clf_config = tree.DecisionTreeClassifier()
                # Entrenamos el clasificador con los embeddings de configuracion y sus labels.
                clf_config.fit(X_train_config, y_train_config)
                # Realizamos la predicción de la parte de test para comprobar acurracy
                prediction_config = clf_config.predict(X_test_config)
                # Obtenemos el acurracy comparando las prediciones y los labels correctos.
                accuracy = accuracy_score(y_test_config, prediction_config)
                print(f"Accuracy clasificador DecisionTree para configuracion: {accuracy:.4%}")
                # Obtencion de las métricas del clasificador para conocer su capacidad de clasificación
                DecisionTreeAccuracy = binario_clasificador.getAccuracy(y_test_config, prediction_config)
                DecisionTreeF1 = binario_clasificador.getF1(y_test_config, prediction_config, "Configuracion")
                DecisionTreeMatthews = binario_clasificador.getMatthews(y_test_config, prediction_config)
                # Guardamos los datos de las métricas obtenidas
                results.append(
                        ["Decision Tree_configuracion", DecisionTreeAccuracy, DecisionTreeMatthews, DecisionTreeF1])
                # Guardamos el clasificador entrenado para clasificar celdas de configuracion
                binario_clasificador.save_model(
                        ruta + "/Modelos_codebert/clasificadores_binarios/DecisionTreeClassifier_config.pkl",
                        clf_config)

                # DecisionTreeClassifier
                # Cargamos el clasificador
                clf_visualizacion = tree.DecisionTreeClassifier()
                # Entrenamos el clasificador con los embeddings de visualizacion y sus labels.
                clf_visualizacion.fit(X_train_visualizacion, y_train_visualizacion)
                # Realizamos predicciones sobre el conjunto de test
                prediction_visualizacion = clf_visualizacion.predict(X_test_visualizacion)
                # Obtenemos el acurracy comparando las prediciones y los labels correctos.
                accuracy = accuracy_score(y_test_visualizacion, prediction_visualizacion)
                print(f"Accuracy clasificador DecisionTree para visualizacion: {accuracy:.4%}")
                # Obtencion de las métricas del clasificador para conocer su capacidad de clasificación
                DecisionTreeAccuracy = binario_clasificador.getAccuracy(y_test_visualizacion, prediction_visualizacion)
                DecisionTreeF1 = binario_clasificador.getF1(y_test_visualizacion, prediction_visualizacion,
                                                            "Visualizacion")
                DecisionTreeMatthews = binario_clasificador.getMatthews(y_test_visualizacion, prediction_visualizacion)
                # Guardamos los datos de las métricas obtenidas
                results.append(
                        ["Decision Tree_visualizacion", DecisionTreeAccuracy, DecisionTreeMatthews, DecisionTreeF1])
                # Guardamos el clasificador entrenado para clasificar celdas de configuracion
                binario_clasificador.save_model(
                        ruta + "/Modelos_codebert/clasificadores_binarios/DecisionTreeClassifier_visualizacion.pkl",
                        clf_visualizacion)

                # DecisionTreeClassifier
                # Cargamos el clasificador
                clf_procesado = tree.DecisionTreeClassifier()
                # Entrenamos el clasificador con los embeddings de procesado y sus labels.
                clf_procesado.fit(X_train_procesado, y_train_procesado)
                # Realizamos predicciones sobre el conjunto de test
                prediction_procesado = clf_procesado.predict(X_test_procesado)
                # Obtenemos el acurracy comparando las prediciones y los labels correctos.
                accuracy = accuracy_score(y_test_procesado, prediction_procesado)
                print(f"Accuracy clasificador DecisionTree para procesado: {accuracy:.4%}")
                # Obtencion de las métricas del clasificador para conocer su capacidad de clasificación
                DecisionTreeAccuracy = binario_clasificador.getAccuracy(y_test_procesado, prediction_procesado)
                DecisionTreeF1 = binario_clasificador.getF1(y_test_procesado, prediction_procesado, "Procesado")
                DecisionTreeMatthews = binario_clasificador.getMatthews(y_test_procesado, prediction_procesado)
                # Guardamos los datos de las métricas obtenidas
                results.append(["Decision Tree_procesado", DecisionTreeAccuracy, DecisionTreeMatthews, DecisionTreeF1])
                # Guardamos el clasificador entrenado para clasificar celdas de configuracion
                binario_clasificador.save_model(
                        ruta + "/Modelos_codebert/clasificadores_binarios/DecisionTreeClassifier_procesado.pkl",
                        clf_procesado)
                # GaussianNB
                clf_config = GaussianNB()
                clf_config.fit(X_train_config, y_train_config)
                prediction_config = clf_config.predict(X_test_config)
                accuracy = accuracy_score(y_test_config, prediction_config)
                print(f"Accuracy clasificador GaussianNB para configuracion: {accuracy:.4%}")
                # Guardamos los resultados del clasificador
                GaussianNBAccuracy = binario_clasificador.getAccuracy(y_test_config, prediction_config)
                GaussianNBF1 = binario_clasificador.getF1(y_test_config, prediction_config, "Configuracion")
                GaussianNBMatthews = binario_clasificador.getMatthews(y_test_config, prediction_config)
                results.append(["GaussianNB_config", GaussianNBAccuracy, GaussianNBMatthews, GaussianNBF1])
                binario_clasificador.save_model(
                        ruta + "/Modelos_codebert/clasificadores_binarios/GaussianNBClassifier_config.pkl",
                        clf_config)

                # GaussianNB
                clf_visualizacion = GaussianNB()
                clf_visualizacion.fit(X_train_visualizacion, y_train_visualizacion)
                prediction_visualizacion = clf_visualizacion.predict(X_test_visualizacion)
                accuracy = accuracy_score(y_test_visualizacion, prediction_visualizacion)
                print(f"Accuracy clasificador GaussianNB para visualizacion: {accuracy:.4%}")
                # Guardamos los resultados del clasificador
                GaussianNBAccuracy = binario_clasificador.getAccuracy(y_test_visualizacion, prediction_visualizacion)
                GaussianNBF1 = binario_clasificador.getF1(y_test_visualizacion, prediction_visualizacion,
                                                          "Visualizacion")
                GaussianNBMatthews = binario_clasificador.getMatthews(y_test_visualizacion, prediction_visualizacion)
                results.append(["GaussianNB_visualizacion", GaussianNBAccuracy, GaussianNBMatthews, GaussianNBF1])
                binario_clasificador.save_model(
                        ruta + "/Modelos_codebert/clasificadores_binarios/GaussianNB_visualizacion.pkl",
                        clf_visualizacion)

                # GaussianNB
                clf_procesado = GaussianNB()
                clf_procesado.fit(X_train_procesado, y_train_procesado)
                prediction_procesado = clf_procesado.predict(X_test_procesado)
                accuracy = accuracy_score(y_test_procesado, prediction_procesado)
                print(f"Accuracy clasificador GaussianNB para procesado: {accuracy:.4%}")
                # Guardamos los resultados del clasificador
                GaussianNBAccuracy = binario_clasificador.getAccuracy(y_test_procesado, prediction_procesado)
                GaussianNBF1 = binario_clasificador.getF1(y_test_procesado, prediction_procesado, "Procesado")
                GaussianNBMatthews = binario_clasificador.getMatthews(y_test_procesado, prediction_procesado)
                results.append(
                        ["GaussianNB_procesado", GaussianNBAccuracy, GaussianNBMatthews, GaussianNBF1])
                binario_clasificador.save_model(
                        ruta + "/Modelos_codebert/clasificadores_binarios/GaussianNB_procesado.pkl",
                        clf_procesado)

                # MLPCLassifier
                clf_config = MLPClassifier(random_state=0, max_iter=300)
                clf_config.fit(X_train_config, y_train_config)
                prediction_config = clf_config.predict(X_test_config)
                accuracy = accuracy_score(y_test_config, prediction_config)
                print(f"Accuracy clasificador MLPClassifier para configuracion: {accuracy:.4%}")
                # Guardamos los resultados del clasificador
                MLPClassifierAccuracy = binario_clasificador.getAccuracy(y_test_config, prediction_config)
                MLPClassifierF1 = binario_clasificador.getF1(y_test_config, prediction_config, "Configuracion")
                MLPClassifierMatthews = binario_clasificador.getMatthews(y_test_config, prediction_config)
                results.append(
                        ["MLPClassifier_configuracion", MLPClassifierAccuracy, MLPClassifierMatthews, MLPClassifierF1])
                binario_clasificador.save_model(
                        ruta + "/Modelos_codebert/clasificadores_binarios/MLPClassifier_config.pkl",
                        clf_config)

                # MLPClassifier
                clf_visualizacion = MLPClassifier(random_state=0, max_iter=300)
                clf_visualizacion.fit(X_train_visualizacion, y_train_visualizacion)
                prediction_visualizacion = clf_visualizacion.predict(X_test_visualizacion)
                accuracy = accuracy_score(y_test_visualizacion, prediction_visualizacion)
                print(f"Accuracy clasificador MLPClassifier para visualizacion: {accuracy:.4%}")
                # Guardamos los resultados del clasificador
                MLPClassifierAccuracy = binario_clasificador.getAccuracy(y_test_visualizacion, prediction_visualizacion)
                MLPClassifierF1 = binario_clasificador.getF1(y_test_visualizacion, prediction_visualizacion,
                                                             "Visualizacion")
                MLPClassifierMatthews = binario_clasificador.getMatthews(y_test_visualizacion, prediction_visualizacion)
                results.append(
                        ["MLPClassifier_visualizacion", MLPClassifierAccuracy, MLPClassifierMatthews, MLPClassifierF1])
                binario_clasificador.save_model(
                        ruta + "/Modelos_codebert/clasificadores_binarios/MLPClassifier_visualizacion.pkl",
                        clf_visualizacion)

                # MLPClassifier
                clf_procesado = MLPClassifier(random_state=0, max_iter=300)
                clf_procesado.fit(X_train_procesado, y_train_procesado)
                prediction_procesado = clf_procesado.predict(X_test_procesado)
                accuracy = accuracy_score(y_test_procesado, prediction_procesado)
                print(f"Accuracy clasificador MLPClassifier para procesado: {accuracy:.4%}")
                # Guardamos los resultados del clasificador
                MLPClassifierAccuracy = binario_clasificador.getAccuracy(y_test_procesado, prediction_procesado)
                MLPClassifierF1 = binario_clasificador.getF1(y_test_procesado, prediction_procesado, "Procesado")
                MLPClassifierMatthews = binario_clasificador.getMatthews(y_test_procesado, prediction_procesado)
                results.append(
                        ["MLPCLassifier_procesado", MLPClassifierAccuracy, MLPClassifierMatthews, MLPClassifierF1])
                binario_clasificador.save_model(
                        ruta + "/Modelos_codebert/clasificadores_binarios/MLPClassifier_procesado.pkl",
                        clf_procesado)

                # RandomForestClassifier
                clf_config = RandomForestClassifier(n_estimators=30, max_depth=4, random_state=1)
                clf_config.fit(X_train_config, y_train_config)
                prediction_config = clf_config.predict(X_test_config)
                accuracy = accuracy_score(y_test_config, prediction_config)
                print(f"Accuracy clasificador RandomForest para configuracion: {accuracy:.4%}")
                # Guardamos los resultados del clasificador
                RandomForestClassifierAccuracy = binario_clasificador.getAccuracy(y_test_config, prediction_config)
                RandomForestClassifierF1 = binario_clasificador.getF1(y_test_config, prediction_config, "Configuracion")
                RandomForestClassifierMatthews = binario_clasificador.getMatthews(y_test_config, prediction_config)
                results.append(["RandomForestClassifier_config", RandomForestClassifierAccuracy,
                                RandomForestClassifierMatthews, RandomForestClassifierF1])
                binario_clasificador.save_model(
                        ruta + "/Modelos_codebert/clasificadores_binarios/RandomForestClassifier_config.pkl",
                        clf_config)

                # RandomForestClassifier
                clf_visualizacion = RandomForestClassifier(n_estimators=30, max_depth=4, random_state=1)
                clf_visualizacion.fit(X_train_visualizacion, y_train_visualizacion)
                prediction_visualizacion = clf_visualizacion.predict(X_test_visualizacion)
                accuracy = accuracy_score(y_test_visualizacion, prediction_visualizacion)
                print(f"Accuracy clasificador RandomForest para visualizacion: {accuracy:.4%}")
                # Guardamos los resultados del clasificador
                RandomForestClassifierAccuracy = binario_clasificador.getAccuracy(y_test_visualizacion,
                                                                                  prediction_visualizacion)
                RandomForestClassifierF1 = binario_clasificador.getF1(y_test_visualizacion, prediction_visualizacion,
                                                                      "Visualizacion")
                RandomForestClassifierMatthews = binario_clasificador.getMatthews(y_test_visualizacion,
                                                                                  prediction_visualizacion)
                results.append(["RandomForestClassifier_visualizacion", RandomForestClassifierAccuracy,
                                RandomForestClassifierMatthews, RandomForestClassifierF1])
                binario_clasificador.save_model(
                        ruta + "/Modelos_codebert/clasificadores_binarios/RandomForestClassifier_visualizacion.pkl",
                        clf_visualizacion)

                # RandomForestClassifier
                clf_procesado = RandomForestClassifier(n_estimators=30, max_depth=4, random_state=1)
                clf_procesado.fit(X_train_procesado, y_train_procesado)
                prediction_procesado = clf_procesado.predict(X_test_procesado)
                accuracy = accuracy_score(y_test_procesado, prediction_procesado)
                print(f"Accuracy clasificador RandomForest para procesado: {accuracy:.4%}")
                # Guardamos los resultados del clasificador
                RandomForestClassifierAccuracy = binario_clasificador.getAccuracy(y_test_procesado,
                                                                                  prediction_procesado)
                RandomForestClassifierF1 = binario_clasificador.getF1(y_test_procesado, prediction_procesado,
                                                                      "Procesado")
                RandomForestClassifierMatthews = binario_clasificador.getMatthews(y_test_procesado,
                                                                                  prediction_procesado)
                results.append(
                        ["RandomForest_procesado", RandomForestClassifierAccuracy, RandomForestClassifierMatthews,
                         RandomForestClassifierF1])
                binario_clasificador.save_model(
                        ruta + "/Modelos_codebert/clasificadores_binarios/RandomForestClassifier_procesado.pkl",
                        clf_procesado)
                # Almaceno los resultados de la precision y matthews de cada clasificador
                resultados = pd.DataFrame(results, columns=["Classifier", "Accuracy", "Matthews", "F1"])
                # Envio los datos a un csv
                resultados.to_csv(ruta + "/Modelos_codebert/Resultados_binario.csv")
                return 0

        def getAccuracy(y_test, predictions):
                """
                Método encargado de calcular la precision de un clasificador en base a sus predicciones.
                Args:
                    predictions: Predicciones realizadas por el clasificador para el conjunto de test

                Returns:
                Devuelve las metricas asociadas al clasificador.
                """
                return sklearn.metrics.accuracy_score(y_test, predictions, normalize=True, sample_weight=None)

        def getF1(y_test, predictions, positivo):
                """
                Método encargado de calcular el valor F1 para un clasificador
                Args:
                    predictions: Predicciones realizadas por el clasificador para el conjunto de test

                Returns:
                El valor calculado para F1
                """
                f1 = f1_score(y_test, predictions, pos_label=positivo)
                return f1

        def getMatthews(y_test, predictions):
                """
                Método encargado de calcular el valor Matthews para un clasificador
                Args:
                    predictions: Predicciones realizadas por el clasificador para el conjunto de test

                Returns:
                 El valor calculado para Matthews
                """

                return sklearn.metrics.matthews_corrcoef(y_test, predictions)

        def save_model(pkl_filename, classifier):
                """
                Método encargado de guardar los modelos de clasificacion
                Args:
                    classifier: Clasificador que se desea guardar
                Returns:0
                """
                with open(pkl_filename, 'wb') as file:
                        pickle.dump(classifier, file)
                return 0

        def clasificacion(ruta, cadena_source):
                """
                Método encargado de aplicar el modelo de clasificacion a las celdas.
                Args:
                    cadena_source: Conjunto de celdas que se van a clasificar

                Returns:
                Diccionario que contiene las clasificaciones asi como su numero de celda.
                """
                # Cargo los clasificadores preentrenados anteriormente
                with open(ruta + "/Modelos_codebert/clasificadores_binarios/RandomForestClassifier_config.pkl", 'rb') as file_config:
                        classifier_config = pickle.load(file_config)
                with open(ruta + "/Modelos_codebert/clasificadores_binarios/RandomForestClassifier_visualizacion.pkl", 'rb') as file_visualizaciones:
                        classifier_visualizaciones = pickle.load(file_visualizaciones)
                with open(ruta + "/Modelos_codebert/clasificadores_binarios/RandomForestClassifier_procesado.pkl", 'rb') as file_procesado:
                        classifier_procesado = pickle.load(file_procesado)
                with open(ruta + "/Modelos_codebert/tokenizer.pkl", 'rb') as file:
                        tokenizer = pickle.load(file)

                #Cargo el modelo para tokenizar
                model = AutoModel.from_pretrained("microsoft/codebert-base")
                visualizaciones = 0
                celdas_visualizaciones = []
                celdas_config = []
                configuracion = 0
                procesado=0
                celdas_procesado=[]
                resultados = {}

                #Recorro cada una de las celdas y el numero de celda al que corresponde
                for i, t in zip(cadena_source['celdas'], cadena_source['num_celdas']):
                        cadena_unida = []
                        cadena_unida.append("".join(i))

                        #Realizo los embeddings de la cadena
                        entities_tokenize = tokenizer(cadena_unida, return_tensors="pt", padding='max_length',
                                                      truncation=True,
                                                      max_length=30, add_special_tokens=True)
                        entities_embed = model(entities_tokenize.input_ids)[0].prod(dim=1).detach().numpy()
                        #Aplico los clasificadores a los embeddings de la celda para predicir el tipo de celda
                        prediccion_config = classifier_config.predict(entities_embed)
                        prediccion_valor= classifier_config.predict_proba(entities_embed)
                        prediccion_visualizaciones= classifier_visualizaciones.predict(entities_embed)
                        prediccion_procesado=classifier_procesado.predict(entities_embed)

                        #Compruebo las predicciones de la celda
                        #Compruebo si es configuracion o noconfiguracion
                        for label in prediccion_config:
                                if (label == 'Configuracion'):
                                        #Aumento la variable de conteo de celdas de configuracion
                                        configuracion += 1
                                        #añado el número de la celda que ha sido clasificada como configuracion al array de configuracion
                                        celdas_config.append(t)

                        #Compruebo si es visualizacion o novisualizacion
                        for label in prediccion_visualizaciones:
                                if (label == 'Visualizacion'):
                                        # Aumento la variable de conteo de celdas de visualizacion
                                        visualizaciones += 1
                                        # añado el número de la celda que ha sido clasificada como visualizacion al array de visualizaciones
                                        celdas_visualizaciones.append(t)

                        #Compruebo si es procesado o noprocesado
                        for label in prediccion_procesado:
                                if (label == 'Procesado'):
                                        # Aumento la variable de conteo de celdas de visualizacion
                                        procesado += 1
                                        # añado el número de la celda que ha sido clasificada como procesado al array de procesado
                                        celdas_procesado.append(t)


                #Creo un diccionario para almacenar los resultados
                #EL diccionario contendrá el numero de total de celdas de cada tipo y por cada tipo el numero de celdas que son de ese tipo.
                resultados['visualizaciones'] = visualizaciones
                resultados['celdas_visualizaciones'] = celdas_visualizaciones
                resultados['configuracion'] = configuracion
                resultados['celdas_configuracion'] = celdas_config
                resultados['procesado'] = procesado
                resultados['celdas_procesado'] = celdas_procesado

                return resultados
