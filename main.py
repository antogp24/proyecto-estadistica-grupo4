import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Variables globales.
CATEGORY_COLUMNS = ['Company', 'TypeName', 'Cpu', 'Gpu',
                    'OpSys', 'ScreenResolution', 'Memory']
NUMERICAL_COLUMNS = ['Inches', 'Ram', 'Price', 'Weight']


def inr_to_usd(inr: float) -> float:
    return inr * 0.012


def mkdir_if_necessary(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory, parents=True)
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


def generate_graph_png(df: pd.DataFrame, column_x: str, column_y: str):
    plt.figure(figsize=[20, 8])
    plt.title(f'{column_x} vs. {column_y}')

    colors = sns.color_palette("husl", len(df[column_x].unique()))
    sns.barplot(
            data=df,
            x=column_x,
            y=column_y,
            hue=column_x,
            palette=colors)

    plt.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.xticks(rotation=90, fontsize=10)

    plt.xlabel(column_x, fontsize=12, weight='bold')
    plt.ylabel(column_y, fontsize=12, weight='bold')

    for p in plt.gca().patches:
        plt.gca().annotate(
            f'{p.get_height():.1f}',
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='center', fontsize=8, color='black',
            weight='bold', xytext=(0, 5), textcoords='offset points')

    png_path: str = f'./images/{column_x}_vs_{column_y}.png'
    if os.path.exists(png_path):
        os.remove(png_path)

    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"Se generó '{png_path}'")


def generate_numerical_vs_categorical_graphs(clean: pd.DataFrame) -> None:
    start_time = time.time()

    sns.set(style="whitegrid", palette="muted")
    for category in CATEGORY_COLUMNS:
        for number in NUMERICAL_COLUMNS:
            generate_graph_png(clean, category, number)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Se guardaron las imágenes en {elapsed_time:.2f} segundos")


def main() -> None:
    unclean: pd.DataFrame = pd.read_csv('laptopData.csv')
    print(f"Número de filas antes de la limpieza: {unclean.shape[0]}")

    clean: pd.DataFrame = clean_laptop_dataset(unclean)
    print(f"Número de filas después de la limpieza: {clean.shape[0]}")

    mkdir_if_necessary("./images/")
    generate_numerical_vs_categorical_graphs(clean)


if __name__ == "__main__":
    main()
