# -*- coding: utf-8 -*-
"""
Simulación de Eventos Discretos — Internación Hospitalaria (TP5)
Metodología: Evento a Evento (TPLL / TPS[i] con HV)
Dataset: Healthcare Dataset (Kaggle) — filtrado por Medical Condition = 'Diabetes'

UTN — FRBA · Simulación
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ============================================================================
# SECCION 1 — PARAMETROS CONFIGURABLES
# ============================================================================
HV = float("inf")                 # High Value: cama libre -> TPS[i] = HV

TF = 365 * 2                      # Tiempo final de simulación (en dias)
WARMUP = 90                       # Período de calentamiento (en dias) que NO se mide
NUM_CORRIDAS = 30                 # Corridas por valor de N
SEMILLA_BASE = 42

MAX_COLA = 10                     # Si hay >= MAX_COLA pacientes esperando, el nuevo se DERIVA
COSTO_CAMA_OCIOSA = 1846.0        # 56% (personal, AHA 2025) de USD 3.297/día (KFF-AHA 2024)
PDER_MAX = 2.0                    # % maximo de derivaciones aceptado 
INGRESO_POR_PACIENTE = None       # USD perdidos por derivacion (None = media de Billing Amount)

RANGO_N = list(range(70, 101, 2)) # Barrido de camas para analisis de sensibilidad
N_ACTUAL = 80                     # Escenario actual 
N_PEOR = 74                       # Escenario peor (recorte de camas)

BASE = os.path.dirname(os.path.abspath(__file__))
RUTA_CSV = os.path.join(BASE, "healthcare_dataset.csv")
RUTA_RES = os.path.join(BASE, "resultados")


# ============================================================================
# SECCION 2 — PROCESAMIENTO DE DATOS Y FDP (continuidad del TP4)
# ============================================================================
def cargar_y_ajustar(ruta_csv):
    df = pd.read_csv(ruta_csv)
    d = df[df["Medical Condition"] == "Diabetes"].copy()
    d["A"] = pd.to_datetime(d["Date of Admission"], errors="coerce")
    d["D"] = pd.to_datetime(d["Discharge Date"], errors="coerce")
    d = d.dropna(subset=["A", "D"]).sort_values("A")

    # --- TI: tiempo de internación (días) ---
    ti = (d["D"] - d["A"]).dt.days
    ti = ti[ti > 0]
    ti_min, ti_max = int(ti.min()), int(ti.max())
    frec = ti.value_counts().sort_index()
    chi_ti = stats.chisquare(frec.values)   # H0: uniforme discreta
    print(f"TI  -> Uniforme discreta [{ti_min}, {ti_max}]  | media={ti.mean():.2f} d"
          f" | Chi2 p-valor={chi_ti.pvalue:.4f}")

    # --- IIA: llegadas ---
    # Las fechas tienen granularidad diaria: muchos pacientes ingresan el mismo día
    # (IIA=0). Por eso se modela la TASA de llegadas por día (proceso de Poisson)
    # y se generan IIA ~ Exponencial(media = 1/lambda).
    dias = pd.date_range(d["A"].min(), d["A"].max())
    llegadas_dia = d.groupby("A").size().reindex(dias, fill_value=0)
    lam = llegadas_dia.mean()
    print(f"IIA -> Exponencial, lambda={lam:.3f} pac/día  | media IIA={24/lam:.2f} h"
          f" | var/media conteos={llegadas_dia.var()/lam:.2f}")

    ingreso = d["Billing Amount"].clip(lower=0).mean()
    return dict(lam=lam, ti_min=ti_min, ti_max=ti_max, ingreso=ingreso,
                ti=ti, llegadas_dia=llegadas_dia)


def graficar_fdp(p, ruta):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ti = p["ti"]
    ax[0].hist(ti, bins=np.arange(p["ti_min"], p["ti_max"] + 2) - 0.5, density=True,
               color="steelblue", edgecolor="white", alpha=.8, label="Datos")
    ax[0].axhline(1 / (p["ti_max"] - p["ti_min"] + 1), color="r", lw=2,
                  label="Uniforme discreta")
    ax[0].set(title="TI — Tiempo de internación", xlabel="Días", ylabel="Probabilidad")
    ax[0].legend()

    c = p["llegadas_dia"]
    k = np.arange(0, c.max() + 1)
    ax[1].bar(k, [np.mean(c == i) for i in k], color="steelblue", alpha=.8, label="Datos")
    ax[1].plot(k, stats.poisson.pmf(k, p["lam"]), "ro-", label=f"Poisson λ={p['lam']:.2f}")
    ax[1].set(title="Llegadas por día (⇒ IIA exponencial)", xlabel="Pacientes/día",
              ylabel="Probabilidad")
    ax[1].legend()
    plt.tight_layout(); plt.savefig(ruta, dpi=150); plt.close()


# ============================================================================
# SECCION 3 — MOTOR DE SIMULACION EVENTO A EVENTO
# ============================================================================
def simular(N, p, semilla=None, tf=TF, warmup=WARMUP, max_cola=MAX_COLA):
    """
    Variables
      Datos:     IIA, TI
      Control:   N (camas), MAXC = max_cola (máx. pacientes esperando cama)
      Estado:    NS (pacientes en el sistema: internados + en espera)
      TEF:       TPLL, TPS[i]
      Resultado: PTO[i], PTO medio, PEC (días), PDER (%), costos
    """
    rng = np.random.default_rng(semilla)
    gen_iia = lambda: rng.exponential(1 / p["lam"])
    gen_ti = lambda: float(rng.integers(p["ti_min"], p["ti_max"] + 1))

    # --- Condiciones iniciales ---
    T = 0.0
    TPLL = 0.0
    TPS = np.full(N, HV)
    ITO = np.zeros(N)            # inicio de tiempo ocioso de cada cama
    STO = np.zeros(N)            # sumatoria de tiempo ocioso de cada cama
    NS = 0
    cola = []                    # tiempos de llegada de pacientes esperando (algoritmo FIFO)

    CLL = 0; CDER = 0; NAT = 0; SEC = 0.0   # contadores medidos post-warmup
                                             #NAT cuenta todos los que ocuparon cama

    while True:
        i = int(np.argmin(TPS))              # prox salida
        if TPLL <= TPS[i]:
            # ================= LLEGADA =================
            T = TPLL
            if T > tf:
                break
            TPLL = T + gen_iia()
            medir = T >= warmup
            if medir:
                CLL += 1
            if NS < N:                       # hay cama libre
                NS += 1
                x = int(np.argmax(TPS == HV))  # busco un puesto libre: TPS(x) = HV
                if medir or ITO[x] >= warmup:
                    STO[x] += T - max(ITO[x], warmup)
                TPS[x] = T + gen_ti()
                if medir:
                    NAT += 1
            elif len(cola) < max_cola:       # espera cama
                NS += 1
                cola.append(T)
            else:                            # derivacion a otro centro 
                if medir:
                    CDER += 1
        else:
            # ================= SALIDA (ALTA) de cama i =================
            T = TPS[i]
            if T > tf:
                break
            NS -= 1                          #act del v de estado
            if cola:                         # entra el primero de la cola
                t_lleg = cola.pop(0)
                TPS[i] = T + gen_ti()
                if t_lleg >= warmup:
                    SEC += T - t_lleg
                    NAT += 1
            else:
                ITO[i] = T
                TPS[i] = HV

    # Ocio que queda al final
    T = tf
    for j in range(N):
        if TPS[j] == HV:
            STO[j] += T - max(ITO[j], warmup)

    horizonte = tf - warmup
    PTO = STO / horizonte * 100
    PTO_medio = PTO.mean()
    PEC = SEC / NAT if NAT else 0.0
    PDER = CDER / CLL * 100 if CLL else 0.0
    costo_ocio = STO.sum() * COSTO_CAMA_OCIOSA
    costo_der = CDER * p["ingreso"]
    return dict(N=N, PTO_medio=PTO_medio, PEC_dias=PEC, PDER=PDER,
                pac_dia=CLL / horizonte, der_mes=CDER / horizonte * 30,
                costo_ocio_mes=costo_ocio / horizonte * 30,
                costo_der_mes=costo_der / horizonte * 30,
                costo_total_mes=(costo_ocio + costo_der) / horizonte * 30)


def ejecutar_escenario(N, p):
    res = pd.DataFrame([simular(N, p, SEMILLA_BASE + k) for k in range(NUM_CORRIDAS)])
    out = res.mean().to_dict()
    for c in ["PTO_medio", "PEC_dias", "PDER", "costo_total_mes"]:
        out[c + "_std"] = res[c].std()
    out["N"] = N
    return out


# ============================================================================
# SECCION 4 — GRAFICOS Y EXPORTACION
# ============================================================================
def graficos(df, n_opt, ruta):
    fig, ax = plt.subplots(1, 4, figsize=(22, 4.8))
    specs = [("PTO_medio", "PTO medio (%)", "seagreen"),
             ("PEC_dias", "Espera promedio en cola (días)", "steelblue"),
             ("PDER", "Pacientes derivados (%)", "coral"),
             ("costo_total_mes", "Costo total (USD/mes)", "slategray")]
    for a, (col, lab, colr) in zip(ax, specs):
        colores = [("gold" if n == n_opt else colr) for n in df["N"]]
        a.bar(df["N"], df[col], color=colores, edgecolor="white")
        a.set(xlabel="Camas (N)", title=lab)
        a.set_xticks(df["N"][::2])
    plt.suptitle("Análisis de sensibilidad — Cantidad de camas", fontweight="bold")
    plt.tight_layout(); plt.savefig(ruta, dpi=150, bbox_inches="tight"); plt.close()


# ============================================================================
# MAIN
# ============================================================================
def main():
    os.makedirs(RUTA_RES, exist_ok=True)
    print("=" * 64 + "\n  SIMULACIÓN EaE — INTERNACIÓN HOSPITALARIA\n" + "=" * 64)

    p = cargar_y_ajustar(RUTA_CSV)
    if INGRESO_POR_PACIENTE is not None:
        p["ingreso"] = INGRESO_POR_PACIENTE
    graficar_fdp(p, os.path.join(RUTA_RES, "fdp_TI_IIA.png"))
    print(f"Ingreso medio por paciente (Billing): USD {p['ingreso']:,.0f}")
    print(f"TF={TF} d | warm-up={WARMUP} d | corridas={NUM_CORRIDAS} | MAX_COLA={MAX_COLA}\n")

    filas = []
    for n in RANGO_N:
        r = ejecutar_escenario(n, p)
        filas.append(r)
        print(f"N={n:3d} | PTO={r['PTO_medio']:5.2f}% | PEC={r['PEC_dias']:5.2f} d | "
              f"PDER={r['PDER']:5.2f}% | Costo/mes=USD {r['costo_total_mes']:,.0f}")
    df = pd.DataFrame(filas)

        # Óptimo = mínimo costo total cumpliendo PDER <= PDER_MAX
    df_ok = df[df["PDER"] <= PDER_MAX]
    n_opt = int(df_ok.loc[df_ok["costo_total_mes"].idxmin(), "N"])
    print(f"\n★ N óptimo (mínimo costo con PDER ≤ {PDER_MAX}%): {n_opt}")

    esc = []
    for nombre, n in [("Peor", N_PEOR), ("Actual", N_ACTUAL), ("Mejor (óptimo)", n_opt)]:
        fila = df[df["N"] == n].iloc[0].to_dict()
        fila["Escenario"] = nombre
        esc.append(fila)
    df_esc = pd.DataFrame(esc)[["Escenario", "N", "PTO_medio", "PEC_dias", "PDER",
                                "der_mes", "costo_ocio_mes", "costo_der_mes",
                                "costo_total_mes"]]
    print("\n" + df_esc.round(2).to_string(index=False))

    with pd.ExcelWriter(os.path.join(RUTA_RES, "resultados_simulacion.xlsx")) as w:
        df_esc.round(2).to_excel(w, sheet_name="Escenarios", index=False)
        df.round(3).to_excel(w, sheet_name="Sensibilidad_N", index=False)
    df.round(3).to_csv(os.path.join(RUTA_RES, "resultados_simulacion.csv"), index=False)
    graficos(df, n_opt, os.path.join(RUTA_RES, "sensibilidad_N.png"))
    print(f"\nResultados guardados en: {RUTA_RES}")


if __name__ == "__main__":
    main()
