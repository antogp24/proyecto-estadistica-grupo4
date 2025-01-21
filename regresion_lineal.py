import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats


def regresion_lineal_anova(clean: pd.DataFrame) -> None:
    # Definir variables
    X = clean['Ram']  # Variable independiente
    y = clean['Price']  # Variable dependiente

    # Agregar constante para el intercepto (β0)
    X = sm.add_constant(X)

    # Ajustar el modelo de regresión
    modelo = sm.OLS(y, X).fit()

    # Coeficientes del modelo
    beta_0 = modelo.params['const']
    beta_1 = modelo.params['Ram']
    print(f"Ecuación de la recta: y = {beta_0:.4f} + {beta_1:.4f}x")

    # Datos necesarios para la tabla ANOVA
    SCT = np.sum((y - y.mean()) ** 2)
    SCR = np.sum((modelo.fittedvalues - y.mean()) ** 2)
    SCE = np.sum((y - modelo.fittedvalues) ** 2)

    glr = 1           # Grados de libertad de la regresión lineal simple
    gle = y.shape[0] - 2  # Grados de libertad del error (n - k, con k=2)
    glt = glr + gle   # Grados de libertad total

    MCR = SCR / glr   # Media cuadrática de la regresión
    MCE = SCE / gle   # Media cuadrática del error

    F_obs = MCR / MCE  # Estadístico F observado

    # Imprimir tabla ANOVA con formato legible
    print("\nTabla ANOVA:")
    print(f"{'Fuentes de Variación':<20}{'Grados de Libertad':<20}{'Sumas Cuadráticas':<20}{'Medias Cuadráticas':<20}{'Estadístico F':<20}")
    print(f"{'Regresión':<20}{glr:<20}{SCR:<20.4f}{MCR:<20.4f}{F_obs:<20.4f}")
    print(f"{'Error':<20}{gle:<20}{SCE:<20.4f}{MCE:<20.4f}{'':<20}")
    print(f"{'Total':<20}{glt:<20}{SCT:<20.4f}{'':<20}{'':<20}")

    # Determinar el valor crítico de F para un nivel de significancia α=0.05
    alpha = 0.05
    F_critico = scipy.stats.f.ppf(1 - alpha, glr, gle)
    print(f"\nEstadístico F observado: {F_obs:.4f}")
    print(f"F crítico para α = {alpha}: {F_critico:.4f}")

    # Conclusión
    if F_obs > F_critico:
        print("Se rechaza H0: La pendiente no es igual a 0.", end=' ')
        print("Existe relación entre RAM y Precio.")
    else:
        print("No se rechaza H0: No hay evidencia suficiente", end=' ')
        print(" para afirmar que la pendiente es diferente de 0.")
