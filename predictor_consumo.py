"""
predictor_consumo.py — modelo global multi-series para la app «Análisis de consumo de agua».

Este módulo se usa EN EL ENTRENAMIENTO (cuaderno de Colab) y EN EL SERVIDOR
(servidor_prediccion.py), de modo que las variables de entrada se construyen
exactamente igual en ambos lados.

Unidad de predicción: CELDA = USO_ESTRATO × COMUNA × SECTOR_HIDRAULICO.
Por celda y mes se usa:
    total        = suma de CONSUMO_DIA de las matrículas con dato   (m³/día)
    n_matriculas = número de matrículas con dato
Estrategia: predicción directa multi-horizonte con un solo regresor
(HistGradientBoostingRegressor) que recibe el horizonte h como variable.
Objetivo normalizado: total[t+h] / escala, con escala = media de los últimos 12 meses.
"""
from __future__ import annotations

import math
import warnings
import numpy as np

VERSION_FEATURES = 1
CATEGORICAS = ("uso_estrato", "comuna", "sector_hidraulico")
SIN_DATO = "(Sin dato)"


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def norm_etiqueta(v) -> str:
    """Etiqueta canónica: sin espacios; vacío → «(Sin dato)»; numéricas en forma mínima.
    Así «2.10», «2.1» y «2.100» (p. ej., un CSV exportado con pandas) son la misma categoría."""
    if v is None:
        return SIN_DATO
    s = str(v).strip()
    if s == "" or s.lower() == "nan":
        return SIN_DATO
    try:
        f = float(s.replace(",", "."))
        if math.isfinite(f):
            return format(f, ".10g")
    except ValueError:
        pass
    return s


def clave_celda(uso, comuna, sector) -> str:
    return f"{norm_etiqueta(uso)}|{norm_etiqueta(comuna)}|{norm_etiqueta(sector)}"


def _alinear(M: np.ndarray, L: int) -> np.ndarray:
    """Toma las últimas L columnas; si hay menos, rellena con NaN a la izquierda."""
    C, T = M.shape
    if T >= L:
        return M[:, T - L:]
    return np.hstack([np.full((C, L - T), np.nan), M])


def nombres_features(L: int) -> list[str]:
    return ([f"lag_{L - 1 - i}" for i in range(L)] +
            ["log_escala", "log1p_n", "ratio_n", "yoy_6m", "pendiente_12m",
             "uso_estrato", "comuna", "sector_hidraulico",
             "h", "mes_obj", "sin_mes", "cos_mes"])


def indices_categoricos(L: int) -> list[int]:
    nf = nombres_features(L)
    return [nf.index(c) for c in CATEGORICAS]


# ---------------------------------------------------------------------------
# Variables de entrada (idénticas en entrenamiento e inferencia)
# ---------------------------------------------------------------------------
def construir_X(Y: np.ndarray, N: np.ndarray, cats: np.ndarray, mes_origen: int, H: int, L: int):
    """
    Y, N        (C, ≥1) historia de total y matrículas hasta el mes de origen t (última columna)
    cats        (C, 3)  códigos numéricos de uso_estrato, comuna, sector (NaN = desconocido)
    mes_origen  1..12   mes calendario de t
    Devuelve X (C*H, F) en orden celda-mayor (celda 0 h=1..H, celda 1 h=1..H, …) y escala (C,).
    """
    Y = _alinear(np.asarray(Y, float), L)
    N = _alinear(np.asarray(N, float), L)
    C = Y.shape[0]
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        escala = np.nanmean(Y[:, -12:], axis=1)
        escala = np.where(np.isfinite(escala) & (escala > 0), escala, np.nan)
        lags = Y / escala[:, None]
        n_last = N[:, -1]
        n_mean = np.nanmean(N[:, -12:], axis=1)
        ratio_n = n_last / np.where(n_mean > 0, n_mean, np.nan)
        yoy = (np.nanmean(Y[:, -6:], axis=1) / np.nanmean(Y[:, -18:-12], axis=1)) if L >= 18 else np.full(C, np.nan)
        a = np.nanmean(Y[:, -6:], axis=1); b = np.nanmean(Y[:, -12:-6], axis=1)
        pendiente = (a - b) / escala
    base = np.column_stack([lags, np.log(escala), np.log1p(n_last), ratio_n, yoy, pendiente, cats])
    X = np.repeat(base, H, axis=0)
    h = np.tile(np.arange(1, H + 1), C)
    mes_obj = ((mes_origen - 1 + h) % 12) + 1
    X = np.column_stack([X, h, mes_obj, np.sin(2 * np.pi * mes_obj / 12), np.cos(2 * np.pi * mes_obj / 12)])
    X[~np.isfinite(X)] = np.nan
    return X.astype(np.float32), escala


def _mapas_normalizados(cat_maps: dict) -> dict:
    return {c: {norm_etiqueta(k): v for k, v in cat_maps[c].items()} for c in CATEGORICAS}


def codificar_cats(atributos: list[dict], cat_maps: dict) -> np.ndarray:
    nm = _mapas_normalizados(cat_maps)
    out = np.full((len(atributos), 3), np.nan)
    for i, a in enumerate(atributos):
        for j, c in enumerate(CATEGORICAS):
            out[i, j] = nm[c].get(norm_etiqueta(a.get(c)), np.nan)
    return out


# ---------------------------------------------------------------------------
# Línea base estacional ingenua (misma del «dummy» de la app)
# ---------------------------------------------------------------------------
def estacional_ingenuo(Y: np.ndarray, h: int) -> np.ndarray:
    Y = np.asarray(Y, float)
    C, T = Y.shape
    ultimo = np.array([r[np.isfinite(r)][-1] if np.isfinite(r).any() else 0.0 for r in Y])
    g = np.ones(C)
    if T >= 18:
        with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            r = np.nanmean(Y[:, -6:], axis=1) / np.nanmean(Y[:, -18:-12], axis=1)
        g = np.where(np.isfinite(r) & (r > 0), np.clip(r, 0.8, 1.25), 1.0)
    out = np.zeros((C, h))
    for k in range(1, h + 1):
        if T >= 12:
            s = Y[:, T - 12 + (k - 1) % 12]
            out[:, k - 1] = np.where(np.isfinite(s), s, ultimo) * g ** math.ceil(k / 12)
        else:
            out[:, k - 1] = ultimo
    return np.maximum(out, 0.0)


# ---------------------------------------------------------------------------
# Inferencia (la usa servidor_prediccion.py)
# ---------------------------------------------------------------------------
def predecir_global(Y, Nm, atributos, meses, h, art, n_trayectorias: int = 20, semilla: int = 12345):
    """
    Y, Nm      (C, T) historia por celda que envía la app (NaN = sin dato)
    atributos  lista de dicts con uso_estrato, comuna, sector_hidraulico
    meses      (T,) mes calendario de cada columna
    h          horizonte pedido (≤ art["H"])
    art        artefacto cargado de modelo_global.joblib
    Devuelve media (C,h), trayectorias (C,S,h) o None, conteo (C,h).
    Las trayectorias de una celda pueden ser negativas; solo sus sumas (series de la app) son interpretables.
    """
    L, Hm = art["L"], art["H"]
    if h > Hm:
        raise ValueError(f"El modelo se entrenó para {Hm} meses; se pidieron {h}.")
    Y = np.asarray(Y, float); Nm = np.asarray(Nm, float)
    C = Y.shape[0]
    cats = codificar_cats(atributos, art["cat_maps"])
    X, escala = construir_X(Y, Nm, cats, int(meses[-1]), Hm, L)
    Z = art["modelo"].predict(X).reshape(C, Hm)[:, :h]
    media = Z * escala[:, None]

    # Celdas fuera del dominio de entrenamiento (uso no visto, sin historia útil) → estacional ingenuo
    usos_ok = {norm_etiqueta(u) for u in art["usos_entrenados"]}
    uso_visto = np.array([norm_etiqueta(a.get("uso_estrato")) in usos_ok for a in atributos])
    respaldo = (~uso_visto) | ~np.isfinite(escala) | ~np.isfinite(media).all(axis=1)
    if respaldo.any():
        media[respaldo] = estacional_ingenuo(Y[respaldo], h)
    media = np.maximum(np.nan_to_num(media), 0.0)

    # Trayectorias: remuestreo de residuos (relativos) de un mismo origen de backtest para todas
    # las celdas → conserva la correlación entre celdas y entre horizontes.
    tray = None
    if n_trayectorias and art.get("residuos") is not None:
        rng = np.random.default_rng(semilla)
        if "_residuos_norm" not in art:          # claves con etiquetas canónicas (una sola vez)
            art["_residuos_norm"] = {clave_celda(*k.split("|")): v for k, v in art["residuos"].items()}
        R = art["_residuos_norm"]                # dict clave_celda -> (O, Hm) residuos relativos
        pool = art["pool_residuos"]              # (B, n, O, Hm) por tamaño de celda
        bordes = art["bordes_escala"]
        O = pool.shape[2]
        factor = float(art.get("factor_residuos", 1.0))   # calibración del intervalo (sección 10 del cuaderno)
        esc = np.where(np.isfinite(escala), escala, np.nan_to_num(media.mean(axis=1)))
        bucket = np.clip(np.searchsorted(bordes, np.log(np.maximum(esc, 1e-9))), 0, pool.shape[0] - 1)
        claves = [clave_celda(a.get("uso_estrato"), a.get("comuna"), a.get("sector_hidraulico")) for a in atributos]
        Rown = np.full((C, O, Hm), np.nan)
        for i, k in enumerate(claves):
            r = R.get(k)
            if r is not None and not respaldo[i]:
                Rown[i] = r
        tray = np.empty((C, n_trayectorias, h))
        origenes = rng.integers(0, O, n_trayectorias)
        for s, o in enumerate(origenes):
            v = Rown[:, o, :h].copy()
            falt = ~np.isfinite(v).any(axis=1)
            if falt.any():   # celdas sin residuo propio: residuo de una celda de tamaño parecido, mismo origen
                v[falt] = pool[bucket[falt], rng.integers(0, pool.shape[1], falt.sum()), o, :h]
            # sin recortar a 0 por celda: el recorte sesga la suma; la app recorta el agregado
            tray[:, s] = media + factor * np.nan_to_num(v) * esc[:, None]

    ult_n = np.array([r[r > 0][-1] if (r > 0).any() else 0.0 for r in np.nan_to_num(Nm)])
    conteo = np.repeat(ult_n[:, None], h, axis=1)
    return media, tray, conteo
