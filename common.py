import os
import matplotlib.pyplot as plt


def plt_generate_graph(prepend_path: str,
                       png_name: str,
                       dpi: float = 300,
                       tight_layout: bool = False) -> None:
    png_path: str = os.path.join(prepend_path, png_name)
    if os.path.exists(png_path):
        os.remove(png_path)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(png_path, dpi=dpi)
    plt.close()
    print(f"Se generó '{png_path}'")
