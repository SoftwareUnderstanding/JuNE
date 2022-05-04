import csv
import sklearn
from sklearn import tree
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import accuracy_score
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle
import pandas as pd

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Clasificadores:
    def tokenizar(ruta):
        """
        Método encargado de tokenizar , realizar los embeddings y entrenar al clasificador
        Returns:0
        """
        # Cargo la base de datos y extraigo los datos
        file = csv.reader(open(ruta+"/Entrenamiento/entrenamiento.csv"), delimiter=';')
        tipo = []
        celda = []
        next(file)
        for line in file:
            tipo.append(line[0])
            celda.append(line[1])

        # Defino el transformado
        demo_vectorizer = CountVectorizer(
            tokenizer=Clasificadores.tokenize,
            binary=True
        )

        # Extraigo de la base de datos los datos para test y entrenamiento
        train_text, test_text, train_labels, test_labels = train_test_split(celda, tipo, stratify=tipo)
        print(f"Training examples: {len(train_text)}, testing examples {len(test_text)}")

        #Tranformo los datos
        real_vectorizer = CountVectorizer(tokenizer=Clasificadores.tokenize, binary=True)
        train_X = real_vectorizer.fit_transform(train_text)
        test_X = real_vectorizer.transform(test_text)

        #Aplico clasificadores
        resultados=[]
        # DecisionTreeClassifier
        classifier = tree.DecisionTreeClassifier()
        classifier.fit(train_X, train_labels)
        predicciones = classifier.predict(test_X)
        DecisionTreeAccuracy = Clasificadores.getAccuracy(test_labels, predicciones)
        DecisionTreeF1 = Clasificadores.getF1(test_labels, predicciones)
        DecisionTreeMatthews = Clasificadores.getMatthews(test_labels, predicciones)
        accuracy = accuracy_score(test_labels, predicciones)
        # Almaceno la precision del clasificador DecisionTreeCLassifier
        resultados.append(["Bag of words","DecisionTreeCLassifier", DecisionTreeAccuracy,DecisionTreeMatthews]+DecisionTreeF1)
        print(f"Accuracy: {accuracy:.4%}")
        # Guardo el modelo utilizado utilizando pickle
        pkl_filename = ruta+"/Modelos/TreeDecision.pkl"
        with open(pkl_filename, 'wb') as file:
            pickle.dump(classifier, file)

        """
        # GaussianNB
        classifier = GaussianNB()
        classifier.fit(train_X, train_labels)
        predicciones = classifier.predict(test_X)
        GaussianNBAccuracy = Clasificadores.getAccuracy(test_labels, predicciones)
        GaussianNBF1 = Clasificadores.getF1(test_labels, predicciones)
        GaussianNBMatthews = Clasificadores.getMatthews(test_labels, predicciones)
        accuracy = accuracy_score(test_labels, predicciones)
        # Almaceno la precision del clasificador LogisticRegression
        resultados.append(["Bag of words","GaussianNB", GaussianNBAccuracy,GaussianNBMatthews]+GaussianNBF1)
        print(f"Accuracy: {accuracy:.4%}")
        # Guardo el modelo utilizado utilizando pickle
        pkl_filename = ruta+"/Modelos/GaussianNB.pkl"
        with open(pkl_filename, 'wb') as file:
            pickle.dump(classifier, file)

        """
        # MLPClassifier
        classifier = MLPClassifier(random_state=0, max_iter=300)
        classifier.fit(train_X, train_labels)
        predicciones = classifier.predict(test_X)
        MLPClassifierAccuracy = Clasificadores.getAccuracy(test_labels, predicciones)
        MLPClassifierF1 = Clasificadores.getF1(test_labels, predicciones)
        MLPClassifierMatthews = Clasificadores.getMatthews(test_labels, predicciones)
        accuracy = accuracy_score(test_labels, predicciones)
        # Almaceno la precision del clasificador MLPClassifer
        resultados.append(["Bag of words", "MLPCLassifier", MLPClassifierAccuracy, MLPClassifierMatthews] + MLPClassifierF1)
        print(f"Accuracy: {accuracy:.4%}")
        # Guardo el modelo utilizado utilizando pickle
        pkl_filename = ruta+"/Modelos/MLPCLassifier.pkl"
        with open(pkl_filename, 'wb') as file:
            pickle.dump(classifier, file)

        # Random Forest Classifier
        classifier = RandomForestClassifier(n_estimators=30, max_depth=4, random_state=1)
        classifier.fit(train_X, train_labels)
        predicciones = classifier.predict(test_X)
        RandomForestClassifierAccuracy = Clasificadores.getAccuracy(test_labels, predicciones)
        RandomForestClassifierF1 = Clasificadores.getF1(test_labels, predicciones)
        RandomForestClassifierMatthews = Clasificadores.getMatthews(test_labels, predicciones)
        accuracy = accuracy_score(test_labels, predicciones)
        # Almaceno la precision del clasificador RandomForestClassifer
        resultados.append(["Bag of words", "RandomForest", RandomForestClassifierAccuracy, RandomForestClassifierMatthews] + RandomForestClassifierF1)
        print(f"Accuracy: {accuracy:.4%}")
        # Guardo el modelo utilizado utilizando pickle
        pkl_filename = ruta+"/Modelos/RandomForestClassifier.pkl"
        with open(pkl_filename, 'wb') as file:
            pickle.dump(classifier, file)

        #Guardo el vectorizador
        with open(ruta+"/Modelos/vectorizer.pkl", 'wb') as file:
            pickle.dump(real_vectorizer, file)

        resultados_final = pd.DataFrame(resultados, columns=["Clasificador","Approach", "Accuracy", "Matthews"] + sorted(set(test_labels)))
        resultados_final.to_csv(ruta + "/Modelos_codebert/Resultadosbagofwords.csv")

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

    def tokenize(sentence):
        punctuation = set(string.punctuation)
        tokens = []
        for token in sentence.split():
            new_token = []
            for character in token:
                if character not in punctuation:
                    new_token.append(character.lower())
            if new_token:
                tokens.append("".join(new_token))
        return tokens



    def clasificacion (ruta, cadena_source):
        """
        Método encargado de ejecutar el clasificador.
        Args:
            cadena_source: Cadena que contiene las celdas a clasificar

        Returns: Diccionario
        Devuelve un diccionario que contiene las predicciones hechas por el clasificador
        """
        # Aplico el modelo
        with open(ruta+"/Modelos/TreeDecision.pkl", 'rb') as file:
            pickle_model = pickle.load(file)
        with open(ruta+"/Modelos/vectorizer.pkl", 'rb') as file:
            vector = pickle.load(file)
        visualizaciones = 0
        imports = 0
        for i in cadena_source:
            cadena_unida = []
            cadena_unida.append("".join(i))
            frases_x = vector.transform(cadena_unida)
            prediccion = pickle_model.predict(frases_x)
            for text, label in zip(cadena_unida, prediccion):
                if (label == 'Import'):
                    imports += 1
                elif (label == 'Visualizacion'):
                    visualizaciones += 1
                else:
                    print("no se ha podido clasificar")

        return visualizaciones, imports

