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
import os
import torch
from sklearn.metrics import f1_score
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class codebert_train:

    def entrenamiento(ruta):
        """
        Método encargado de realizar la lectura de la base de entrenamiento , tokenizacion ,
        embeddings y entrenamiento de los clasificadores.
        Returns:0
        """

        # Cargamos la base de datos que vamos a usar para entrenar
        file = csv.reader(open(ruta+"/Entrenamiento/entrenamiento.csv"), delimiter=';')
        labels = []
        celda = []
        next(file)

        #Recorremos el csv y guardamos los valores
        for line in file:
            labels.append(line[0])
            celda.append(line[1])

        #Definimos el tokenizer y el modelo que vamos a usar
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        model = AutoModel.from_pretrained("microsoft/codebert-base")
        model.to(device)
        #Realizamos los embeddings de la cadena celda
        entities = celda
        entities_tokenize = tokenizer(entities, return_tensors="pt", padding='max_length', truncation=True,
                                      max_length=30, add_special_tokens=True)
        entities_embed = model(entities_tokenize.input_ids)[0].prod(dim=1).detach().numpy()
        entity_classes = labels

        #Separamos en entrenamiento y test en un 70/30
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(entities_embed, entity_classes,
                                                                                    train_size=0.7)
        # Variable donde vamos a almacenar los resultados de los clasificadores
        results = []

        #Aplicamos los modelos ( entrenamiento y prediccion)
        #DecisionTreeClassifier
        clf = tree.DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        prediction = clf.predict(X_test)
        #Guardamos los resultados del clasificador
        DecisionTreeAccuracy = codebert_train.getAccuracy(y_test, prediction)
        DecisionTreeF1 = codebert_train.getF1(y_test, prediction)
        DecisionTreeMatthews = codebert_train.getMatthews(y_test, prediction)
        results.append(["Multiclase","Decision Tree", DecisionTreeAccuracy, DecisionTreeMatthews,DecisionTreeF1])
        #Guardamos el modelo
        codebert_train.save_model(ruta+"/Modelos_codebert/DecisionTreeClassifier.pkl", clf)

        #GaussianNB
        clf = GaussianNB()
        clf.fit(X_train, y_train)
        GaussianNBPredictions = clf.predict(X_test)
        #Guardamos los resultados del clasificador
        GaussianNBAccuracy = codebert_train.getAccuracy(y_test, GaussianNBPredictions)
        GaussianNBF1 = codebert_train.getF1(y_test, GaussianNBPredictions)
        GaussianNBMatthews = codebert_train.getMatthews(y_test, GaussianNBPredictions)
        results.append(["Multiclase","GaussianNB", GaussianNBAccuracy, GaussianNBMatthews,GaussianNBF1])
        #Guardamos el modelo
        codebert_train.save_model(ruta + "/Modelos_codebert/GaussianNB.pkl", clf)

        #MLPClassifier
        clf = MLPClassifier(random_state=0, max_iter=300)
        clf.fit(X_train, y_train)
        MLPClassifierPredictions = clf.predict(X_test)
        # Guardamos los resultados del clasificador
        MLPClassifierAccuracy = codebert_train.getAccuracy(y_test, MLPClassifierPredictions)
        MLPClassifierF1 = codebert_train.getF1(y_test, MLPClassifierPredictions)
        MLPClassifierMatthews = codebert_train.getMatthews(y_test, MLPClassifierPredictions)
        results.append(["Multiclase","MLPClassifier", MLPClassifierAccuracy, MLPClassifierMatthews,MLPClassifierF1])
        # Guardamos el modelo
        codebert_train.save_model(ruta + "/Modelos_codebert/MLPClassifier.pkl", clf)

        #RandomForest Classifier
        clf = RandomForestClassifier(n_estimators=30, max_depth=4, random_state=1)
        clf.fit(X_train, y_train)
        RandomForestClassifierPredictions = clf.predict(X_test)
        # Guardamos los resultados del clasificador
        RandomForestClassifierAccuracy = codebert_train.getAccuracy(y_test, RandomForestClassifierPredictions)
        RandomForestClassifierF1 = codebert_train.getF1(y_test, RandomForestClassifierPredictions)
        RandomForestClassifierMatthews = codebert_train.getMatthews(y_test, RandomForestClassifierPredictions)
        results.append(["Multiclase","RandomForestClassifier", RandomForestClassifierAccuracy,
                        RandomForestClassifierMatthews,RandomForestClassifierF1])
        # Guardamos el modelo
        codebert_train.save_model(ruta + "/Modelos_codebert/RandomForest.pkl", clf)

        resultados=pd.DataFrame(results, columns=["Clasificador","Approach", "Accuracy", "Matthews","F1"])
        resultados.to_csv(ruta + "/Modelos_codebert/Resultados_multiclasificador.csv")
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

    def save_model(pkl_filename,classifier):
        """
        Método encargado de guardar los modelos de clasificacion
        Args:
            classifier: Clasificador que se desea guardar
        Returns:0
        """
        with open(pkl_filename, 'wb') as file:
            pickle.dump(classifier, file)
        return 0

    def clasificacion (ruta, cadena_source):
        """
        Método encargado de aplicar el modelo de clasificacion a las celdas.
        Args:
            cadena_source: Conjunto de celdas que se van a clasificar

        Returns:
        Diccionario que contiene las clasificaciones asi como su numero de celda.
        """
        # Aplico el modelo
        with open(ruta+"/Modelos_codebert/DecisionTreeClassifier.pkl", 'rb') as file:
            classifier = pickle.load(file)
        with open(ruta+"/Modelos_codebert/tokenizer.pkl", 'rb') as file:
            tokenizer = pickle.load(file)
        #with open(ruta+"/src/Modelos_codebert/model.pkl", 'rb') as file:
            #model = pickle.load(file)
        model = AutoModel.from_pretrained("microsoft/codebert-base")
        visualizaciones =0
        celdas_visualizaciones=[]
        celdas_config=[]
        celdas_procesado=[]
        config = 0
        procesado=0
        resultados={}

        for i,t in zip(cadena_source['celdas'],cadena_source['num_celdas']):
            cadena_unida = []
            cadena_unida.append("".join(i))
            #print(cadena_unida)
            #prueba
            entities_tokenize=tokenizer(cadena_unida, return_tensors="pt", padding='max_length', truncation=True,
                                      max_length=30, add_special_tokens=True)
            entities_embed= model(entities_tokenize.input_ids)[0].prod(dim=1).detach().numpy()
            prediccion = classifier.predict(entities_embed)
            for label in prediccion:
                if (label == 'Configuracion'):
                    config += 1
                    celdas_config.append(t)
                elif (label == 'Visualizacion'):
                    visualizaciones += 1
                    celdas_visualizaciones.append(t)
                elif (label == 'Procesado'):
                    procesado += 1
                    celdas_procesado.append(t)
                else:
                    print("no se ha podido clasificar")

        resultados['visualizaciones']=visualizaciones
        resultados['celdas_visualizaciones']=celdas_visualizaciones
        resultados['configuracion'] = config
        resultados['celdas_configuracion'] = celdas_config
        resultados['procesado']=procesado
        resultados['celdas_procesado']=celdas_procesado
        return resultados