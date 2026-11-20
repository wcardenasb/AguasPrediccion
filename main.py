"""Aplicación de escritorio Windows: pywebview y modelo global, sin servidor HTTP."""
from __future__ import annotations

import base64
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import webview
from sklearn.ensemble import HistGradientBoostingRegressor  # Facilita la detección por PyInstaller
from predictor_consumo import predecir_global
from svg_pdf_rapido import generar_pdf_vectorial


def recursos() -> Path:
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))


class API:
    def __init__(self):
        self.art = None

    def predecir(self, solicitud: dict) -> dict:
        try:
            if self.art is None:
                self.art = joblib.load(recursos() / "modelo_global.joblib")
            celdas = solicitud.get("series", [])
            if not celdas:
                raise ValueError("No se recibieron celdas para la predicción.")
            h = int(solicitud["horizon"])
            L = len(celdas[0]["total"])
            if L <= 0 or h <= 0:
                raise ValueError("El historial y el horizonte deben ser positivos.")
            if any(len(c["total"]) != L or len(c["n_matriculas"]) != L for c in celdas):
                raise ValueError("Las celdas no tienen la misma cantidad de meses.")
            Y = np.asarray([[np.nan if v is None else float(v) for v in c["total"]] for c in celdas], dtype=float)
            N = np.asarray([[np.nan if v is None else float(v) for v in c["n_matriculas"]] for c in celdas], dtype=float)
            atributos = [{k: c.get(k) for k in ("uso_estrato", "comuna", "sector_hidraulico", "n_registros")} for c in celdas]
            _, mm = map(int, solicitud["history_start"].split("-"))
            if not 1 <= mm <= 12:
                raise ValueError("Mes de inicio inválido.")
            meses = np.array([(mm - 1 + i) % 12 + 1 for i in range(L)], dtype=int)
            media, tray, conteo = predecir_global(Y, N, atributos, meses, h, self.art, n_trayectorias=20)
            resultado = []
            for i, celda in enumerate(celdas):
                total = {"mean": np.round(media[i], 6).tolist()}
                if tray is not None:
                    total["samples"] = np.round(tray[i], 4).tolist()
                row = {"id": str(celda["id"]), "total": total}
                if conteo is not None:
                    row["n_matriculas"] = {"mean": np.round(conteo[i], 3).tolist()}
                resultado.append(row)
            return {"model": "Modelo global entrenado", "series": resultado}
        except Exception as exc:
            logging.exception("Fallo de predicción")
            raise RuntimeError(f"No se pudo ejecutar el modelo: {exc}") from exc

    def _dialogo(self, nombre: str, extension: str, filtro: str):
        rutas = webview.windows[0].create_file_dialog(webview.FileDialog.SAVE, save_filename=nombre, file_types=(filtro,))
        if not rutas:
            return None
        ruta = Path(rutas[0] if isinstance(rutas, (tuple, list)) else rutas)
        return ruta if ruta.suffix.lower() == extension else ruta.with_suffix(extension)

    def guardar_csv(self, nombre, contenido):
        ruta = self._dialogo(nombre, ".csv", "Archivos CSV (*.csv)")
        if ruta is None:
            return {"guardado": False}
        ruta.write_text(contenido, encoding="utf-8-sig")
        return {"guardado": True, "ruta": str(ruta)}

    def guardar_png(self, nombre, imagen_base64):
        ruta = self._dialogo(nombre, ".png", "Imágenes PNG (*.png)")
        if ruta is None:
            return {"guardado": False}
        ruta.write_bytes(base64.b64decode(imagen_base64, validate=True))
        return {"guardado": True, "ruta": str(ruta)}

    def guardar_pdf(self, nombre, svg_texto, titulo, subtitulo):
        ruta = self._dialogo(nombre, ".pdf", "Documentos PDF (*.pdf)")
        if ruta is None:
            return {"guardado": False}
        try:
            inicio = time.perf_counter()
            ruta.write_bytes(generar_pdf_vectorial(svg_texto, titulo, subtitulo))
            print(f"PDF guardado en {time.perf_counter() - inicio:.3f} s: {ruta}", flush=True)
            return {"guardado": True, "ruta": str(ruta)}
        except Exception as exc:
            logging.exception("Error al exportar PDF")
            raise RuntimeError(f"No se pudo guardar el PDF: {exc}") from exc


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    html = recursos() / "consumo_aguasyaguas.html"
    webview.create_window("Aguas y Aguas | Predicción de consumo", html.as_uri(), js_api=API(), width=1280, height=850, min_size=(950, 650))
    webview.start(gui="edgechromium", debug=False)
