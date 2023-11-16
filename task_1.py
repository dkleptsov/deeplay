import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier 


TRAIN_PATH = 'data/task1_train_v4.csv'
TEST_PATH = 'data/task1_test_v4.csv'
PREDICTIONS_FOLDER = 'answer/'
SCORING = 'matthews_corrcoef' #'roc_auc'
CV_FOLDS = 5
CATEGORICAL_FEATURES = ['stat_2']
LGBM_PARAM_GRID = {'num_leaves': [31, 63, 127],
                   'learning_rate': [0.05, 0.1, 0.2],
                   'n_estimators': [150, 175, 200],
                   'min_child_samples': [10, 20, 30],
                   'reg_alpha': [0, 1, 10],
                   'reg_lambda': [0, 1, 10],
                   'verbose':[-1]}
XGB_PARAM_GRID = {'learning_rate': [0.05, 0.1, 0.2],
                   'max_depth': [3, 5, 7],
                   'n_estimators': [50, 100, 200],
                   'subsample': [0.5, 0.7, 1.0],
                   'colsample_bytree': [0.5, 0.7, 1.0],
                   'gamma': [0, 0.1, 0.2],}
LGBM_BEST_PARAMS = {'learning_rate': 0.05,
                    'min_child_samples': 30,
                    'n_estimators': 150,
                    'num_leaves': 127,
                    'reg_alpha': 1,
                    'reg_lambda': 0,
                    'verbose':-1,
                    'objective': 'binary'}
XGB_BEST_PARAMS = {'colsample_bytree': 0.7,
                   'gamma': 0,
                   'learning_rate': 0.05,
                   'max_depth': 7,
                   'n_estimators': 200,
                   'subsample': 0.7}
CLASSIFIERS_NAN_TOLERANT = [LGBMClassifier(**LGBM_BEST_PARAMS),
                            XGBClassifier(**XGB_BEST_PARAMS),
                            HistGradientBoostingClassifier(),
                            DecisionTreeClassifier(),
                            DummyClassifier(strategy="most_frequent"),]
CLASSIFIERS_NAN_INTOLERANT = [
                            RandomForestClassifier(),
                            GradientBoostingClassifier(),
                            KNeighborsClassifier(),
                            GaussianNB(),
                            LogisticRegression(),]
BEST_CLASSIFIER = LGBMClassifier(**LGBM_BEST_PARAMS)


def explore_data(data:pd.DataFrame) -> None:
    """ Проведем исследовательский анализ данных """
    print(f"Размерность данных:\n{data.shape}")
    print(f"\nТипы данных:\n{data.dtypes}")
    print(f"\nОписательная статистика:\n{data.describe()}")
    print(f"\nПервые строчки:\n{data.head()}")
    print(f"\nКоличество уникальных значений:\n{data.nunique()}")
    print(f"\nКоличество пропущенных значений:\n{data.isnull().sum()}")
    print(f"\nРаспределение по классам:\n{data['y'].value_counts()}\n")


def replace_missing_with_avg(df:pd.DataFrame) -> pd.DataFrame:
    """ Заменим пропущенные значения средними значениями по столбцам. """
    for col in df.columns:
        if df[col].dtype != 'object':
            avg = df[col].mean()
            df[col].fillna(avg, inplace=True)
    return df


def replace_missing_with_normal(data:pd.DataFrame) -> pd.DataFrame:
    """
    Заменим пропущенные значения нормальным распределением по столбцам.
    """
    for col in data.columns:
        if data[col].dtype != 'object':
            mean = data[col].mean()
            std = data[col].std()
            null_count = data[col].isnull().sum()
            if null_count > 0:
                null_idx = data[col].isnull()
                data.loc[null_idx, col] = np.random.normal(loc=mean,
                                                           scale=std,
                                                           size=null_count)
    return data


def detect_and_remove_outliers(data:pd.DataFrame) -> pd.DataFrame:
    """ Распознаем и удалим аутлаеры """
    len_before = data.shape[0]  # Сохраним количество строк до удаления
    for col in data.columns:
        if data[col].dtype != 'object':
            z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
            data = data[z_scores <= 3]
    print(f"Удалено {len_before - data.shape[0]} строк с аутлаерами")
    return data


def split_xy(data:pd.DataFrame) -> pd.DataFrame:
    """ Разделяем фичи, результаты """
    X = data.drop('y', axis=1)
    y = data['y']
    return X, y


def encode_categorical_data(X_train:pd.DataFrame,
                            X_test:pd.DataFrame,
                            categorical_features:list) -> tuple:
    """
    Закодируем категориальные признаки с помощью LabelEncoder из scikit-learn.
    """
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()

    for feature in categorical_features:
        X_train_encoded = X_train_encoded.astype({feature: str})
        X_test_encoded = X_test_encoded.astype({feature: str})
        le = LabelEncoder()
        le.fit(X_train[feature])
        X_train_encoded[feature] = le.transform(X_train[feature])
        X_test_encoded[feature] = le.transform(X_test[feature])
    
    return X_train_encoded, X_test_encoded


def scale_data(X_train:pd.DataFrame,
               X_test:pd.DataFrame,
               categorical_features:list) -> pd.DataFrame:
    """ Нормализуем фичи, отдельно трейн и тест чтобы избежать "утечек" """
    scaler = StandardScaler()
    numerical_features = X_train.columns.difference(categorical_features)
    X_train[numerical_features]=scaler.fit_transform(X_train[numerical_features])
    X_test[numerical_features]=scaler.transform(X_test[numerical_features])
    return X_train, X_test


def select_boosting_params(X:pd.DataFrame,
                           y:pd.Series,
                           param_grid:dict, classifier) -> dict:
    """ Подбираем параметры c помощью GridSearch и кроссвалидации """
    grid_search = GridSearchCV(classifier,
                               param_grid,
                               cv=CV_FOLDS,
                               scoring=SCORING,
                               n_jobs=-1,
                               verbose=1)
    grid_search.fit(X, y)
    print(f"The best parameters for {classifier.__class__.__name__}:\
                                  \n{grid_search.best_params_}\n")
    return grid_search.best_params_


def plot_roc_auc(classifiers:list, X:pd.DataFrame, y:pd.Series, title:str)->None:
    """ Построим ROC кривые для всех моделей """
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k--')
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.1,
                                                        random_state=42)
    for classifier in classifiers:
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred)
        label = f'{classifier.__class__.__name__} (AUC = {auc_score:.3f})'
        plt.plot(fpr, tpr, label=label)

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(title)
    plt.legend()
    plt.savefig(f"{PREDICTIONS_FOLDER}task1 {title}.png")
    plt.show()


def test_classifiers(classifiers:list, X:pd.DataFrame, y:pd.Series,) -> dict:
    """ Проверим качество моделей с помощью кроссвалидации """
    scorings = {}
    for classifier in classifiers:
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        n_scores = cross_val_score(classifier, X, y, 
                                   scoring=SCORING, 
                                   cv=cv,
                                   n_jobs=-1,
                                   error_score='raise')
        scorings[classifier.__class__.__name__] = np.mean(n_scores)
    return scorings


def make_barplot(results:dict, title:str) -> None:
    """ Построим barplot c результата классификации """
    df = pd.Series(results).sort_values(ascending=True, inplace=False)
    plt.figure(figsize=(12, 6))
    plt.barh(df.index, df.values, align='center', alpha=0.8, left=0.17)
    plt.title(title)
    plt.savefig(f"{PREDICTIONS_FOLDER}task1 {title}.png")
    plt.show()


def fit_and_predict(classifier,
                    X:pd.DataFrame,
                    y:pd.Series,
                    X_test:pd.DataFrame) -> None:
    """ Обучим модель train данных и сделаем предсказания """
    classifier.fit(X, y)
    y_pred = classifier.predict(X_test)
    y_pred = pd.DataFrame(y_pred, columns=['y'])
    predictions_path = f"{PREDICTIONS_FOLDER}task1_predictions.csv"
    y_pred.to_csv(predictions_path, index=False)
    print(f"Predictions saved to {predictions_path}")


def main():
    #  Загрузим данные
    data = pd.read_csv(TRAIN_PATH, sep=";")
    X_to_predict = pd.read_csv(TEST_PATH, sep=";")

    # Проведем исследовательский анализ данных
    explore_data(data)
    """
    Основные выводы:
    - Все остальные колонки имеют числовой тип данных. 
    - Данные между классами распределены почти поровну (1-7506, 0-7494)
    - Во всех столбцах более 17% (2683/15000) пропусков.
    - Есть незначительное количество (61) строк с аутлаерами.
    - Среднее сильно отличается между колонкам, поэтому необходима нормализация. 
    - Значения во колонке stat_2 принадлежат к небольшому набору (46).
      Значения целочисленные. Это может говорить о том, что это категориальный признак.
    """

    # Разделим фичи и результаты
    X, y = split_xy(data)

    # Закодируем категориальные признаки
    X, X_to_predict = encode_categorical_data(X, X_to_predict,
                                              CATEGORICAL_FEATURES)

    # Нормализуем фичи
    X, X_to_predict = scale_data(X, X_to_predict, CATEGORICAL_FEATURES)

    # #  Подберем и сохраним в виде констант параметры для бустингов
    # lgbm_params = select_boosting_params(X, y, LGBM_PARAM_GRID, LGBMClassifier())
    # xgb_params = select_boosting_params(X, y, XGB_PARAM_GRID, XGBClassifier())

    # Нарисуем ROC кривые для моделей без замены NaN
    plot_roc_auc(CLASSIFIERS_NAN_TOLERANT, X, y,
                 title="ROC AUC без замены NaN средними")

    # Проверим качество моделей с помощью f1 метрики и кроссвалидации
    scorings = test_classifiers(CLASSIFIERS_NAN_TOLERANT, X, y)

    # Построим barplot c результата классификации
    make_barplot(scorings, f"{SCORING} метрика моделей без замены NaN средними")

    # Предобработаем данные
    # Заменим пропущенные значения средними значениями по столбцам.
    data_preprocessed = replace_missing_with_avg(data)

    # Удалим аутлаеры
    data_preprocessed = detect_and_remove_outliers(data_preprocessed)

    # Разделим фичи и результаты    
    X_preprocessed, y_preprocessed = split_xy(data_preprocessed)

    # Нарисуем ROC кривые для моделей с предобработкой данных    
    plot_roc_auc(CLASSIFIERS_NAN_INTOLERANT, X_preprocessed, y_preprocessed,
                 title="ROC AUC с заменой NaN средними")

    # Проверим качество моделей с помощью f1 метрики и кроссвалидации
    scorings = test_classifiers(CLASSIFIERS_NAN_INTOLERANT, X_preprocessed, y_preprocessed )

    # Построим barplot c результата классификации
    make_barplot(scorings, f"{SCORING} метрика моделей с заменой NaN средними")

    # Предскажем результаты для тестовых данных
    # fit_and_predict(BEST_CLASSIFIER, X, y, X_to_predict)


if __name__ == "__main__":
    main()


# TODO:
# Сделать версии для запуска из Java
# Писать на самих барплотах скор
# Поправить имена моделей в барплотах
# Сделать 9 графиков на одной картинке для задания 2
# Проверять существование папок перед сохранения файлов
# Написать тесты

# DOCS:
# https://biodatamining.biomedcentral.com/articles/10.1186/s13040-023-00322-4