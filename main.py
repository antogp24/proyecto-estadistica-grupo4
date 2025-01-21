import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from common import plt_generate_graph
from pruebas import (prueba_hipotesis_1,
                     prueba_hipotesis_2,
                     prueba_hipotesis_3)
from regresion_lineal import regresion_lineal_anova

# Variables globales.
CATEGORY_COLUMNS = ['Company', 'TypeName', 'Cpu', 'Gpu',
                    'OpSys', 'ScreenResolution', 'Memory']

NUMERICAL_COLUMNS = ['Inches', 'Ram', 'Price', 'Weight']

NUMERICAL_EXCEPT_PRICE = NUMERICAL_COLUMNS.copy()
NUMERICAL_EXCEPT_PRICE.remove('Price')

LINEAR_REGRESSION_PATH = './images/simple-linear-regression/'
BOXPLOTS_PATH = './images/boxplots/'
BAR_CHARTS_PATH = './images/versus/'

ANALISIS_DESCRIPTIVO = False
PRUEBAS_DE_HIPOTESIS = False
REGRESION_LINEAL = True


def inr_to_usd(inr: float) -> float:
    return inr * 0.012


def mkdir_if_necessary(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Se creó el directorio {directory}")
    else:
        print(f"Ya existe el directorio {directory}")


def clean_laptop_dataset(unclean: pd.DataFrame) -> pd.DataFrame:
    df: pd.DataFrame = unclean.copy()
    df.drop('Unnamed: 0', axis=1, inplace=True)  # Remover columna 'Unnamed: 0'
    df.dropna(inplace=True)  # Remover todas las filas con valores nulos.
    df = df[(df != '?').all(axis=1)]  # Remover todas las filas que tienen '?'.

    # Quitar los sufijos y convertir a número las siguientes celdas:
    df['Inches'] = df['Inches'].apply(lambda cell: float(cell))
    df['Weight'] = df['Weight'].apply(lambda cell: float(cell.strip('kg')))
    df['Ram'] = df['Ram'].apply(lambda cell: float(cell.strip('GB')))

    # Convertir los precios de rupias indias a dólares.
    df['Price'] = df['Price'].apply(inr_to_usd)

    df.reset_index()
    return df


def get_correlation_matrix(clean: pd.DataFrame) -> pd.DataFrame:
    corr: pd.DataFrame = clean[NUMERICAL_COLUMNS].corr()
    for column in NUMERICAL_COLUMNS:
        series = corr[column]
        corr[column] = series[series != 1]
    return corr


def get_valid_max(corr: pd.DataFrame, column: str) -> (str, float):
    correlations = corr[column]
    correlations = correlations.dropna()
    max_column = correlations.idxmax()
    max_value = correlations.max()
    return max_column, float(max_value)


def generate_boxplot_png(path: str, df: pd.DataFrame, column: str) -> None:
    plt.figure(figsize=[10, 6])
    plt.title(f'Boxplot of {column}', fontsize=16, weight='bold')

    sns.set(style="whitegrid", palette="muted")
    sns.boxplot(data=df, x=column)

    plt.xlabel(column, fontsize=12, weight='bold')
    plt.ylabel('Values', fontsize=12, weight='bold')

    plt.xticks(rotation=45, fontsize=10)
    plt.tight_layout()

    plt_generate_graph(path, f'{column}_boxplot.png', dpi=300)


def generate_graph_png(path: str, df: pd.DataFrame, col_x: str, col_y: str):
    plt.figure(figsize=[20, 8])
    plt.title(f'{col_x} vs. {col_y}')

    colors = sns.color_palette("husl", len(df[col_x].unique()))
    sns.barplot(
            data=df,
            x=col_x,
            y=col_y,
            hue=col_x,
            palette=colors)

    plt.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.xticks(rotation=90, fontsize=10)

    plt.xlabel(col_x, fontsize=12, weight='bold')
    plt.ylabel(col_y, fontsize=12, weight='bold')

    for p in plt.gca().patches:
        plt.gca().annotate(
            f'{p.get_height():.1f}',
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='center', fontsize=8, color='black',
            weight='bold', xytext=(0, 5), textcoords='offset points')

    plt_generate_graph(path,
                       f'{col_x}_vs_{col_y}.png',
                       dpi=300,
                       tight_layout=True)


def generate_numerical_vs_categorical_graphs(path: str, clean: pd.DataFrame):
    sns.set(style="whitegrid", palette="muted")
    for category in CATEGORY_COLUMNS:
        for number in NUMERICAL_COLUMNS:
            generate_graph_png(path, clean, category, number)


def generate_numerical_boxplots(path: str, clean: pd.DataFrame) -> None:
    for column in NUMERICAL_COLUMNS:
        generate_boxplot_png(path, clean, column)


def generate_X_vs_Price(path: str, clean: pd.DataFrame, X: str) -> None:
    plt.scatter(clean[X], clean['Price'])

    plt.xlabel(f'{X}')
    plt.ylabel('Price (USD)')
    plt.title(f'{X} vs. Price')

    plt_generate_graph(path, f'{X}_vs_Price.png', dpi=300)


def main() -> None:
    unclean: pd.DataFrame = pd.read_csv('laptopData.csv')
    clean: pd.DataFrame = clean_laptop_dataset(unclean)

    print(f"Número de filas antes de la limpieza: {unclean.shape[0]}")
    print(f"Número de filas después de la limpieza: {clean.shape[0]}\n")

    mkdir_if_necessary("./images/")

    if ANALISIS_DESCRIPTIVO:
        mkdir_if_necessary(BOXPLOTS_PATH)
        mkdir_if_necessary(BAR_CHARTS_PATH)

        start_time: float = time.time()

        generate_numerical_vs_categorical_graphs(BAR_CHARTS_PATH, clean)
        generate_numerical_boxplots(BOXPLOTS_PATH, clean)

        end_time: float = time.time()
        total_time: float = end_time - start_time
        print(f"Se guardaron las imágenes en {total_time:.2f} segundos")

    if PRUEBAS_DE_HIPOTESIS:
        prueba_hipotesis_1(clean)
        prueba_hipotesis_2(clean)
        prueba_hipotesis_3(clean)

    if REGRESION_LINEAL:
        mkdir_if_necessary(LINEAR_REGRESSION_PATH)

        corr = get_correlation_matrix(clean)
        print(corr)
        max_category, max_corr = get_valid_max(corr, 'Price')
        print(f'La variable {max_category} tiene el mayor coeficiente', end='')
        print(f'de correlación lineal con Price, con un valor de {max_corr}\n')

        for column in NUMERICAL_EXCEPT_PRICE:
            generate_X_vs_Price(LINEAR_REGRESSION_PATH, clean, column)

        regresion_lineal_anova(clean)

    return None


if __name__ == "__main__":
    main()
