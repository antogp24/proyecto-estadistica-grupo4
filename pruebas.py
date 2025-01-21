import pandas as pd
import numpy as np
from scipy import stats


def prueba_hipotesis_1(clean):
    print("Prueba de Hipótesis 1")
    print("-----------------------------------------------------------")

    def filter_gpu_price(clean, name: str):
        data = clean[clean['Gpu'].str.contains(name, case=False, na=False)]
        return data['Price']

    # Filtrar precios según GPU
    nvidia_prices = filter_gpu_price(clean, "Nvidia")
    intel_prices = filter_gpu_price(clean, "Intel")

    # Calcular estadísticas muestrales (Nvidia)
    n1 = len(nvidia_prices)
    media1 = np.mean(nvidia_prices)
    s1 = np.std(nvidia_prices, ddof=1)

    # Calcular estadísticas muestrales (Intel)
    n2 = len(intel_prices)
    media2 = np.mean(intel_prices)
    s2 = np.std(intel_prices, ddof=1)

    print("Datos:")
    print(f"\tNvidia: n1={n1}, media1={media1:.2f}, s1={s1:.2f}")
    print(f"\tIntel:  n2={n2}, media2={media2:.2f}, s2={s2:.2f}")

    # Prueba de igualdad de varianzas
    F_obs, p_value_var = stats.levene(nvidia_prices, intel_prices)
    alpha = 0.05
    equal_var = p_value_var > alpha
    print("Prueba de varianzas:")
    print(f"\tF_obs={F_obs:.2f}")
    print(f"\tp-value={p_value_var:.4f}")
    print(f"\tp-value > {alpha} == {equal_var}")
    print(f"\t\tLas varianzas son {'iguales' if equal_var else 'distintas'}")

    # Selección del estadístico (varianzas iguales o desiguales)
    t_obs, p_value = stats.ttest_ind(
            nvidia_prices,
            intel_prices,
            equal_var=equal_var,
            alternative='greater')

    # Grados de libertad.
    v = None
    if equal_var:
        v = n1 + n2 - 2
    else:
        p = (s1**2/n1 + s2**2/n2)**2
        q = (s1**2/n1)**2 / (n1-1) + (s2**2/n2)**2 / (n2-1)
        v = p / q

    # Región crítica
    alpha = 0.05
    t_crit = stats.t.ppf(1 - alpha, df=v)

    print(f"Estadístico de prueba:\n\tt_obs={t_obs:.2f}")
    print("Región Crítica:")
    print(f"\tgrados-de-libertad={v}")
    print(f"\tt_crit={t_crit:.2f}")
    print(f"\tC(x)=[x: t_obs > {t_crit:.2f}]")
    print("Conclusión:")
    if t_obs > t_crit:
        print("\tTenemos evidencia para rechazar H0, por tanto las laptops")
        print("\tcon GPUs de Nvidia son más caras que las laptops con GPUs")
        print("\tde Intel.")
    else:
        print("\tTenemos evidencia para no rechazar H0, por tanto las laptops")
        print("\tcon GPUs de Nvidia son menor o igual de caras que las")
        print("\tlaptops con GPUs de Intel.")


def prueba_hipotesis_2(clean):
    print("Prueba de Hipótesis 2")
    print("-----------------------------------------------------------")

    def filter_memory_price(clean, name: str):
        pattern = r'^[^+]*' + name + r'[^+]*$'
        data = clean[clean['Memory'].str.match(pattern, case=False, na=False)]
        return data['Price']

    # Filtrar tipos de almacenamiento
    only_ssd_prices = filter_memory_price(clean, "SSD")
    only_hdd_prices = filter_memory_price(clean, "HDD")

    n = clean.shape[0]
    n1 = only_ssd_prices.shape[0]
    n2 = only_hdd_prices.shape[0]

    # Proporciones
    p1 = n1 / n
    p2 = n2 / n

    print(f"SSD: n1={n1}, p1={p1:.4f}")
    print(f"HDD: n2={n2}, p2={p2:.4f}")

    # Estadístico de prueba para diferencia de proporciones
    p = (n1 * p1 + n2 * p2) / (n1 + n2)
    Z_obs = (p1 - p2) / np.sqrt(p * (1 - p) * (2 / n))

    # Región crítica
    alpha = 0.05
    Z_crit = stats.norm.ppf(1 - alpha)

    print(f"Estadístico de prueba:\n\tZ_obs={Z_obs:.2f}")
    print("Región crítica:")
    print(f"\tZ_crit={Z_crit:.2f}")
    print(f"\tC(x)=[x: Z_obs > {Z_crit:.2f}]")
    print("Conclusión:")
    if Z_obs > Z_crit:
        print("\tTenemos evidencia para rechazar H0, por tanto la proporción")
        print("\tde laptops con sólo SSD es mayor que la proporción de")
        print("\tlaptops con sólo HDD.")
    else:
        print("\tTenemos evidencia para no rechazar H0, por tanto la")
        print("\tproporción de laptops con sólo SSD es menor o igual que")
        print("\tla proporción de laptops con sólo HDD.")


def prueba_hipotesis_3(clean):
    print("Prueba de Hipótesis 3")
    print("-----------------------------------------------------------")

    # Tabla de contingencia
    table = pd.crosstab(clean['OpSys'], clean['TypeName'])
    print("Tabla de contingencia:")
    print(table)

    # Prueba de independencia (Chi-cuadrado)
    alpha = 0.05
    chi2_obs, p_value, dof, expected = stats.chi2_contingency(table)
    chi2_crit = stats.chi2.ppf(1 - alpha, df=dof)

    print(f"Estadístico de prueba:\n\tchi2_obs={chi2_obs:.2f}")
    print("Región crítica:")
    print(f"\tgrados-de-libertad={dof}")
    print(f"\tvalor-crítico={chi2_crit:.2f}")
    print(f"\tC(x)=[x: chi2_obs > {chi2_crit:.2f}]")

    print("Conclusión:")
    if chi2_obs > chi2_crit:
        print("\tTenemos evidencia para rechazar H0, por tanto el sistema")
        print("\toperativo es independiente del tipo de laptop.")
    else:
        print("\tTenemos evidencia para no rechazar H0, por tanto el")
        print("\tsistema operativo es dependiente del tipo de laptop.")
