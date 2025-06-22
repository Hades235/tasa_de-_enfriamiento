import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

class CuerpoEnfriado:
    def __init__(self, T0, T_amb, T_obs, t_obs):
        # Variables numéricas
        self.T0 = T0                 # Temperatura inicial del objeto
        self.Tam = T_amb             # Temperatura ambiente
        self.T_obs = T_obs           # Temperatura observada en cierto tiempo
        self.t_obs = t_obs           # Tiempo en que se observó T_obs

        # Variables simbólicas
        self.T0_sym = sp.Symbol('T0')        # Temperatura inicial (simbólica)
        self.Tamb_sym = sp.Symbol('Tamb')    # Temperatura ambiente (simbólica)
        self.k = sp.Symbol('k')              # Constante de proporcionalidad
        self.t = sp.Symbol('t')              # Tiempo (variable simbólica)

        # Ecuación de enfriamiento simbólica de Newton
        self.T_expr = self.Tamb_sym + (self.T0_sym - self.Tamb_sym) * sp.exp(-self.k * self.t)

        # Derivadas simbólicas
        self.dT_dt = sp.diff(self.T_expr, self.t)         # Primera derivada
        self.d2T_dt2 = sp.diff(self.dT_dt, self.t)        # Segunda derivada

    def mostrar_derivadas(self):
        print("\nEcuación simbólica de enfriamiento:")
        sp.pprint(self.T_expr)
        print("\nPrimera derivada (tasa de cambio):")
        sp.pprint(self.dT_dt)
        print("\nSegunda derivada (concavidad):")
        sp.pprint(self.d2T_dt2)

    def calcular_k(self):
        """Calcula la constante de proporcionalidad k a partir de datos observados."""
        if self.T_obs is None or self.t_obs is None:
            raise ValueError("Faltan datos experimentales para calcular la variable k.")
        fraccion = (self.T_obs - self.Tam) / (self.T0 - self.Tam)
        k_val = -(1 / self.t_obs) * np.log(fraccion)
        return round(k_val, 4)

    def evaluar_temperatura(self, k_val, tiempo):
        """Evalúa la función T(t) en forma numérica para un array de tiempo."""
        T_func = lambda t: self.Tam + (self.T0 - self.Tam) * np.exp(-k_val * t)
        return [T_func(ti) for ti in tiempo]

    def graficar(self, k_val, minutos=60):
        """Genera la gráfica de temperatura en función del tiempo."""
        tiempos = np.linspace(0, minutos, 300)
        temperaturas = self.evaluar_temperatura(k_val, tiempos)

        plt.figure(figsize=(8, 5))
        plt.plot(tiempos, temperaturas, label='Temperatura del objeto', color='blue')
        plt.axhline(self.Tam, linestyle='--', color='gray', label='Temperatura ambiente')
        plt.title('Enfriamiento de un cuerpo (Ley de Newton)')
        plt.xlabel('Tiempo (minutos)')
        plt.ylabel('Temperatura (°C)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

# ========================
# USO DEL PROGRAMA
# ========================

if __name__ == "__main__":
    # Crear una instancia con datos reales
    cuerpo = CuerpoEnfriado(T0=90, T_amb=20, T_obs=60, t_obs=10)

    # Mostrar derivadas simbólicas
    cuerpo.mostrar_derivadas()

    # Calcular constante k
    k_valor = cuerpo.calcular_k()
    print(f"\nConstante k calculada: {k_valor}")

    # Graficar temperatura en función del tiempo
    cuerpo.graficar(k_valor)

