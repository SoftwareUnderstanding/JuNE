import pickle
import csv
from transformers import AutoTokenizer, AutoModel, TrainingArguments, AutoModelForPreTraining, AutoModelForMaskedLM
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import sklearn
from sklearn.svm import SVC, LinearSVC
from sklearn import linear_model

class binario_clasificador:

        def embeddings(ruta):
                file_configuracion = csv.reader(open(ruta + "/Entrenamiento/Configuracion.csv"), delimiter=';')
                labels_configuracion = []
                celda_configuracion = []
                next(file_configuracion)
                # Recorremos el csv y guardamos los valores
                print("hola")
                for line in file_configuracion:
                        labels_configuracion.append(line[0])
                        celda_configuracion.append(line[1])

                print(len(labels_configuracion))
                print(len(celda_configuracion))

                file_visualizacion= csv.reader(open(ruta + "/Entrenamiento/Visualizacion.csv"), delimiter=';')
                labels_visualizacion = []
                celda_visualizacion= []
                next(file_visualizacion)
                # Recorremos el csv y guardamos los valores

                for line in file_visualizacion:
                        labels_visualizacion.append(line[0])
                        celda_visualizacion.append(line[1])

                print(len(labels_visualizacion))
                print(len(celda_visualizacion))
                file_procesado= csv.reader(open(ruta + "/Entrenamiento/Procesado.csv"), delimiter=';')
                labels_procesado = []
                celda_procesado= []
                next(file_procesado)

                # Recorremos el csv y guardamos los valores
                for line in file_procesado:
                        labels_procesado.append(line[0])
                        celda_procesado.append(line[1])



                # Definimos el tokenizer y el modelo que vamos a usar
                tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
                model = AutoModel.from_pretrained("microsoft/codebert-base")

                # Realizamos los embeddings de la cadenas de configuracion
                entities_configuracion = celda_configuracion
                entities_tokenize_configuracion = tokenizer(entities_configuracion, return_tensors="pt", padding='max_length', truncation=True,
                                              max_length=12, add_special_tokens=True)
                entities_embed_configuracion = model(entities_tokenize_configuracion.input_ids)[0].prod(dim=1).detach().numpy()
                entity_classes_configuracion = labels_configuracion

                # Realizamos los embeddings de las cadenas de visualizacion
                entities_visualizacion = celda_visualizacion
                entities_tokenize_visualizacion= tokenizer(entities_visualizacion, return_tensors="pt",
                                                            padding='max_length', truncation=True,
                                                            max_length=12, add_special_tokens=True)
                entities_embed_visualizacion = model(entities_tokenize_visualizacion.input_ids)[0].prod(
                        dim=1).detach().numpy()
                entity_classes_visualizacion= labels_configuracion

                # Realizamos los embeddings de la cadenas de procesado
                entities_procesado= celda_procesado
                entities_tokenize_procesado= tokenizer(entities_procesado, return_tensors="pt",
                                                            padding='max_length', truncation=True,
                                                            max_length=12, add_special_tokens=True)
                entities_embed_procesado= model(entities_tokenize_procesado.input_ids)[0].prod(
                        dim=1).detach().numpy()
                entity_classes_procesado = labels_procesado

                print("hago embeddings")

                return entities_embed_configuracion, entity_classes_configuracion, entities_embed_visualizacion,\
                       entity_classes_visualizacion, entities_embed_procesado, entity_classes_procesado


        def entrenamiento(ruta,entities_embed_configuracion, entity_classes_configuracion, entities_embed_visualizacion,
                       entity_classes_visualizacion, entities_embed_procesado, entity_classes_procesado):

                # Separamos en entrenamiento y test en un 70/30
                X_train_config, X_test_config, y_train_config, y_test_config = sklearn.model_selection.train_test_split(entities_embed_configuracion,
                                                                                            entity_classes_configuracion,
                                                                                            train_size=0.7)

                X_train_visualizacion, X_test_visualizacion, y_train_visualizacion, y_test_visualizacion = sklearn.model_selection.train_test_split(
                        entities_embed_visualizacion,
                        entity_classes_visualizacion,
                        train_size=0.7)

                X_train_procesado, X_test_procesado, y_train_procesado, y_test_procesado= sklearn.model_selection.train_test_split(
                        entities_embed_procesado,
                        entity_classes_procesado,
                        train_size=0.7)

                # Variable donde vamos a almacenar los resultados de los clasificadores
                results = []
                # Aplicamos los modelos ( entrenamiento y prediccion)
                # DecisionTreeClassifier
                clf_config = tree.DecisionTreeClassifier()
                clf_config.fit(X_train_config, y_train_config)
                prediction_config = clf_config.predict(X_test_config)
                accuracy = accuracy_score(y_test_config, prediction_config)
                print(f"Accuracy: {accuracy:.4%}")
                # Guardamos los resultados del clasificador
                DecisionTreeAccuracy = binario_clasificador.getAccuracy(y_test_config, prediction_config)
                DecisionTreeF1 = binario_clasificador.getF1(y_test_config, prediction_config)
                DecisionTreeMatthews = binario_clasificador.getMatthews(y_test_config, prediction_config)
                results.append(["Decision Tree_config", DecisionTreeAccuracy, DecisionTreeMatthews] + DecisionTreeF1)
                binario_clasificador.save_model(ruta + "/Modelos_codebert/DecisionTreeClassifier_config.pkl", clf_config)

                print("guardo modelo")

                # DecisionTreeClassifier
                clf_visualizacion = tree.DecisionTreeClassifier()
                clf_visualizacion.fit(X_train_visualizacion, y_train_visualizacion)
                prediction_visualizacion= clf_visualizacion.predict(X_test_visualizacion)
                accuracy = accuracy_score(y_test_visualizacion, prediction_visualizacion)
                print(f"Accuracy: {accuracy:.4%}")
                # Guardamos los resultados del clasificador
                DecisionTreeAccuracy = binario_clasificador.getAccuracy(y_test_visualizacion, prediction_visualizacion)
                DecisionTreeF1 = binario_clasificador.getF1(y_test_visualizacion, prediction_visualizacion)
                DecisionTreeMatthews = binario_clasificador.getMatthews(y_test_visualizacion, prediction_visualizacion)
                results.append(["Decision Tree_visualizacion", DecisionTreeAccuracy, DecisionTreeMatthews] + DecisionTreeF1)
                binario_clasificador.save_model(ruta + "/Modelos_codebert/DecisionTreeClassifier_visualizacion.pkl",
                                                clf_visualizacion)

                # DecisionTreeClassifier
                clf_procesado= tree.DecisionTreeClassifier()
                clf_procesado.fit(X_train_procesado, y_train_procesado)
                prediction_procesado = clf_procesado.predict(X_test_procesado)
                accuracy = accuracy_score(y_test_procesado, prediction_procesado)
                print(f"Accuracy: {accuracy:.4%}")
                # Guardamos los resultados del clasificador
                DecisionTreeAccuracy = binario_clasificador.getAccuracy(y_test_procesado, prediction_procesado)
                DecisionTreeF1 = binario_clasificador.getF1(y_test_procesado, prediction_procesado)
                DecisionTreeMatthews = binario_clasificador.getMatthews(y_test_procesado, prediction_procesado)
                results.append(
                        ["Decision Tree_visualizacion", DecisionTreeAccuracy, DecisionTreeMatthews] + DecisionTreeF1)
                binario_clasificador.save_model(ruta + "/Modelos_codebert/DecisionTreeClassifier_procesado.pkl",
                                                clf_procesado)

                #resultados = pd.DataFrame(results, columns=["Approach", "Accuracy", "Matthews"] + sorted(set(y_test)))
                #resultados.to_csv(ruta + "/src/Modelos_codebert/Resultados.csv")
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

        def getF1(y_test, predictions):
                """
                Método encargado de calcular el valor F1 para un clasificador
                Args:
                    predictions: Predicciones realizadas por el clasificador para el conjunto de test

                Returns:
                El valor calculado para F1
                """
                f1 = []
                for l in sorted(set(y_test)):
                        f1.append(sklearn.metrics.f1_score(y_test, predictions, labels=[l], average='micro'))
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
                # Aplico el modelo
                with open(ruta + "/Modelos_codebert/DecisionTreeClassifier_config.pkl", 'rb') as file_config:
                        classifier_config = pickle.load(file_config)
                with open(ruta + "/Modelos_codebert/DecisionTreeClassifier_visualizacion.pkl", 'rb') as file_visualizaciones:
                        classifier_visualizaciones = pickle.load(file_visualizaciones)
                with open(ruta + "/Modelos_codebert/DecisionTreeClassifier_procesado.pkl", 'rb') as file_procesado:
                        classifier_procesado = pickle.load(file_procesado)
                with open(ruta + "/Modelos_codebert/tokenizer.pkl", 'rb') as file:
                        tokenizer = pickle.load(file)

                model = AutoModel.from_pretrained("microsoft/codebert-base")
                visualizaciones = 0
                celdas_visualizaciones = []
                celdas_config = []
                configuracion = 0
                procesado=0
                celdas_procesado=[]
                resultados = {}

                for i, t in zip(cadena_source['celdas'], cadena_source['num_celdas']):
                        cadena_unida = []
                        cadena_unida.append("".join(i))
                        # print(cadena_unida)
                        # prueba
                        entities_tokenize = tokenizer(cadena_unida, return_tensors="pt", padding='max_length',
                                                      truncation=True,
                                                      max_length=12, add_special_tokens=True)
                        entities_embed = model(entities_tokenize.input_ids)[0].prod(dim=1).detach().numpy()
                        prediccion_config = classifier_config.predict(entities_embed)
                        prediccion_valor= classifier_config.predict_proba(entities_embed)
                        prediccion_visualizaciones= classifier_visualizaciones.predict(entities_embed)
                        prediccion_procesado=classifier_procesado.predict(entities_embed)
                        for label in prediccion_config:
                                if (label == 'Configuracion'):
                                        configuracion += 1
                                        celdas_config.append(t)


                        for label in prediccion_visualizaciones:
                                if (label == 'Visualizacion'):
                                        visualizaciones += 1
                                        celdas_visualizaciones.append(t)

                        for label in prediccion_procesado:
                                if (label == 'Procesado'):
                                        procesado += 1
                                        celdas_procesado.append(t)



                resultados['visualizaciones'] = visualizaciones
                resultados['celdas_visualizaciones'] = celdas_visualizaciones
                resultados['configuracion'] = configuracion
                resultados['celdas_configuracion'] = celdas_config
                resultados['procesado'] = procesado
                resultados['celdas_procesado'] = celdas_procesado

                return resultados
