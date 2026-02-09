import math
import os
import re
from urllib.parse import parse_qs, urlparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def entropia_shannon(texto: str) -> float:
    if not texto:
        return 0.0
    conteos = {}
    for caracter in texto:
        conteos[caracter] = conteos.get(caracter, 0) + 1
    n = len(texto)
    entropia = 0.0
    for c in conteos.values():
        p = c / n
        entropia -= p * math.log2(p)
    return entropia


def entropia_relativa(texto: str) -> float:
    if not texto or len(texto) <= 1:
        return 0.0
    ent = entropia_shannon(texto)
    denominador = math.log2(len(texto))
    return ent / denominador if denominador else 0.0


_acortadores = {
    "bit.ly",
    "tinyurl.com",
    "goo.gl",
    "t.co",
    "ow.ly",
    "is.gd",
    "buff.ly",
    "adf.ly",
    "bit.do",
    "cutt.ly",
    "rebrand.ly",
    "t.ly",
    "shorturl.at",
    "soo.gd",
    "qr.ae",
}

_palabras_sospechosas = [
    "login",
    "signin",
    "verify",
    "verification",
    "update",
    "security",
    "account",
    "bank",
    "paypal",
    "apple",
    "google",
    "microsoft",
    "confirm",
    "webscr",
    "secure",
    "invoice",
    "billing",
    "password",
    "recover",
    "reset",
    "support",
]


def parsear_url(url: str):
    try:
        parsed = urlparse(url)
    except Exception:
        return "", "", "", ""
    esquema = (parsed.scheme or "").lower()
    netloc = (parsed.netloc or "").lower()
    ruta = parsed.path or ""
    consulta = parsed.query or ""
    if "@" in netloc:
        netloc = netloc.split("@")[-1]
    host = netloc.split(":")[0]
    return esquema, host, ruta, consulta


def es_ip(host: str) -> int:
    if not host:
        return 0
    v4 = re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", host)
    if not v4:
        return 0
    partes = host.split(".")
    try:
        ok = all(0 <= int(x) <= 255 for x in partes)
    except Exception:
        return 0
    return 1 if ok else 0


def obtener_tld(host: str) -> str:
    if not host or "." not in host:
        return ""
    return host.rsplit(".", 1)[-1]


def contar_subdominios(host: str) -> int:
    if not host:
        return 0
    partes = host.split(".")
    return max(0, len(partes) - 2)


def longitud_token_mas_largo(texto: str) -> int:
    if not texto:
        return 0
    tokens = re.split(r"[^A-Za-z0-9]+", texto)
    tokens = [t for t in tokens if t]
    return max((len(t) for t in tokens), default=0)


def contar_palabras_sospechosas(texto: str) -> int:
    if not texto:
        return 0
    low = texto.lower()
    return sum(1 for w in _palabras_sospechosas if w in low)


def ratio_seguro(numerador: float, denominador: float) -> float:
    return float(numerador) / float(denominador) if denominador else 0.0


def extraer_caracteristicas(url: str) -> dict:
    u = str(url) if url is not None else ""
    esquema, host, ruta, consulta = parsear_url(u)
    tld = obtener_tld(host)
    completo = u.lower()

    digitos = sum(ch.isdigit() for ch in u)
    letras = sum(ch.isalpha() for ch in u)
    especiales = len(u) - digitos - letras

    caracteristicas = {}
    caracteristicas["longitud_url"] = len(u)
    caracteristicas["longitud_host"] = len(host)
    caracteristicas["longitud_ruta"] = len(ruta)
    caracteristicas["longitud_consulta"] = len(consulta)
    caracteristicas["longitud_tld"] = len(tld)
    caracteristicas["cantidad_puntos"] = u.count(".")
    caracteristicas["cantidad_guiones"] = u.count("-")
    caracteristicas["cantidad_guion_bajo"] = u.count("_")
    caracteristicas["cantidad_arroba"] = u.count("@")
    caracteristicas["cantidad_interrogacion"] = u.count("?")
    caracteristicas["cantidad_igual"] = u.count("=")
    caracteristicas["cantidad_ampersand"] = u.count("&")
    caracteristicas["cantidad_slash"] = u.count("/")
    caracteristicas["cantidad_porcentaje"] = u.count("%")
    caracteristicas["cantidad_digitos"] = digitos
    caracteristicas["cantidad_letras"] = letras
    caracteristicas["cantidad_especiales"] = especiales
    caracteristicas["ratio_digitos"] = ratio_seguro(digitos, len(u))
    caracteristicas["ratio_especiales"] = ratio_seguro(especiales, len(u))
    caracteristicas["tiene_https"] = 1 if esquema == "https" else 0
    caracteristicas["tiene_ip"] = es_ip(host)
    caracteristicas["cantidad_subdominios"] = contar_subdominios(host)
    caracteristicas["tiene_puerto"] = 1 if re.search(r":[0-9]+$", (urlparse(u).netloc or "")) else 0
    caracteristicas["palabras_sospechosas"] = contar_palabras_sospechosas(completo)
    caracteristicas["tiene_punycode"] = 1 if "xn--" in host else 0
    caracteristicas["tiene_redireccion_doble_slash"] = 1 if re.search(r"https?://[^/]+//", u) else 0
    caracteristicas["tiene_http_en_ruta"] = 1 if "http" in (ruta.lower() if ruta else "") else 0
    caracteristicas["cantidad_parametros"] = len(parse_qs(consulta)) if consulta else 0
    caracteristicas["token_mas_largo"] = longitud_token_mas_largo(u)
    caracteristicas["entropia_shannon"] = entropia_shannon(u)
    caracteristicas["entropia_relativa"] = entropia_relativa(u)
    caracteristicas["es_acortador"] = 1 if host in _acortadores else 0
    caracteristicas["tiene_marca_en_host"] = 1 if contar_palabras_sospechosas(host) > 0 else 0
    caracteristicas["tiene_prefijo_sufijo_host"] = 1 if "-" in host else 0
    caracteristicas["cantidad_www"] = host.count("www")
    caracteristicas["cantidad_digitos_host"] = sum(ch.isdigit() for ch in host)
    caracteristicas["ratio_digitos_host"] = ratio_seguro(caracteristicas["cantidad_digitos_host"], len(host))
    caracteristicas["cantidad_puntos_host"] = host.count(".")
    return caracteristicas


def remover_caracteristicas_correlacionadas(datos_x: pd.DataFrame, umbral: float = 0.95):
    correlacion = datos_x.corr(numeric_only=True).abs()
    superior = correlacion.where(np.triu(np.ones(correlacion.shape), k=1).astype(bool))
    a_eliminar = [col for col in superior.columns if any(superior[col] > umbral)]
    return datos_x.drop(columns=a_eliminar), a_eliminar


def evaluar_modelo(nombre: str, modelo, paquete_entrenamiento, paquete_validacion, x_prueba, y_prueba, prefijo_salida: str):
    x_entrenamiento, y_entrenamiento = paquete_entrenamiento
    x_validacion, y_validacion = paquete_validacion

    modelo.fit(x_entrenamiento, y_entrenamiento)

    prob_validacion = modelo.predict_proba(x_validacion)[:, 1]
    pred_validacion = (prob_validacion >= 0.5).astype(int)

    prob_prueba = modelo.predict_proba(x_prueba)[:, 1]
    pred_prueba = (prob_prueba >= 0.5).astype(int)

    matriz_validacion = confusion_matrix(y_validacion, pred_validacion)
    matriz_prueba = confusion_matrix(y_prueba, pred_prueba)

    precision_validacion = precision_score(y_validacion, pred_validacion, zero_division=0)
    recall_validacion = recall_score(y_validacion, pred_validacion, zero_division=0)
    fpr_val, tpr_val, _ = roc_curve(y_validacion, prob_validacion)
    auc_validacion = auc(fpr_val, tpr_val)

    precision_prueba = precision_score(y_prueba, pred_prueba, zero_division=0)
    recall_prueba = recall_score(y_prueba, pred_prueba, zero_division=0)
    fpr_test, tpr_test, _ = roc_curve(y_prueba, prob_prueba)
    auc_prueba = auc(fpr_test, tpr_test)

    plt.figure()
    plt.plot(fpr_val, tpr_val)
    plt.xlabel("Tasa de Falsos Positivos (FPR)")
    plt.ylabel("Tasa de Verdaderos Positivos (TPR)")
    plt.title(f"Curva ROC (Validación) - {nombre} (AUC={auc_validacion:.4f})")
    plt.tight_layout()
    plt.savefig(f"{prefijo_salida}_{nombre}_roc_validacion.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(fpr_test, tpr_test)
    plt.xlabel("Tasa de Falsos Positivos (FPR)")
    plt.ylabel("Tasa de Verdaderos Positivos (TPR)")
    plt.title(f"Curva ROC (Prueba) - {nombre} (AUC={auc_prueba:.4f})")
    plt.tight_layout()
    plt.savefig(f"{prefijo_salida}_{nombre}_roc_prueba.png", dpi=200)
    plt.close()

    print(f"\n=== {nombre} ===")
    print("VALIDACIÓN")
    print("Matriz de confusión:\n", matriz_validacion)
    print("Precisión:", float(precision_validacion))
    print("Recall:", float(recall_validacion))
    print("AUC:", float(auc_validacion))
    print("PRUEBA")
    print("Matriz de confusión:\n", matriz_prueba)
    print("Precisión:", float(precision_prueba))
    print("Recall:", float(recall_prueba))
    print("AUC:", float(auc_prueba))

    return {
        "nombre": nombre,
        "matriz_validacion": matriz_validacion,
        "precision_validacion": float(precision_validacion),
        "recall_validacion": float(recall_validacion),
        "auc_validacion": float(auc_validacion),
        "matriz_prueba": matriz_prueba,
        "precision_prueba": float(precision_prueba),
        "recall_prueba": float(recall_prueba),
        "auc_prueba": float(auc_prueba),
        "modelo": modelo,
    }


def resolver_ruta_dataset():
    candidatos = [
        "dataset_pishing.csv",
        "dataset_phishing.csv",
        os.path.join(os.path.dirname(__file__), "dataset_pishing.csv"),
        os.path.join(os.path.dirname(__file__), "dataset_phishing.csv"),
    ]
    for c in candidatos:
        if os.path.exists(c):
            return c
    raise FileNotFoundError("No se encontró el dataset. Coloca dataset_pishing.csv en la misma carpeta del script.")


def main():
    ruta_csv = resolver_ruta_dataset()
    datos = pd.read_csv(ruta_csv)

    print(datos.head(5))
    if "status" not in datos.columns or "url" not in datos.columns:
        raise ValueError("El CSV debe tener columnas: url, status")

    conteo_estados = datos["status"].value_counts(dropna=False)
    print("\nConteo por estado:\n", conteo_estados)

    datos = datos.dropna(subset=["url", "status"]).copy()

    mapa_estado = {
        "legitimate": 0,
        "legit": 0,
        "phishing": 1,
        "pishing": 1,
    }
    datos["estado_binario"] = datos["status"].astype(str).str.strip().str.lower().map(mapa_estado)
    datos = datos.dropna(subset=["estado_binario"]).copy()
    datos["estado_binario"] = datos["estado_binario"].astype(int)

    datos = datos.drop_duplicates(subset=["url", "estado_binario"]).copy()

    matriz_caracteristicas = pd.DataFrame(list(datos["url"].apply(extraer_caracteristicas)))
    etiquetas = datos["estado_binario"].copy()

    matriz_caracteristicas = matriz_caracteristicas.replace([np.inf, -np.inf], np.nan).fillna(0)

    selector_varianza = VarianceThreshold(threshold=0.0)
    matriz_varianza = selector_varianza.fit_transform(matriz_caracteristicas)
    columnas_mantenidas = matriz_caracteristicas.columns[selector_varianza.get_support()].tolist()
    matriz_varianza = pd.DataFrame(matriz_varianza, columns=columnas_mantenidas)

    matriz_seleccionada, columnas_correlacionadas = remover_caracteristicas_correlacionadas(matriz_varianza, umbral=0.95)

    print("\nColumnas seleccionadas:")
    print(matriz_seleccionada.columns.tolist())
    print("\nColumnas eliminadas (correlacionadas):")
    print(columnas_correlacionadas)

    x_temp, x_prueba, y_temp, y_prueba = train_test_split(
        matriz_seleccionada, etiquetas, test_size=0.30, random_state=42, stratify=etiquetas
    )
    proporcion_val_desde_temp = 0.15 / 0.70
    x_entrenamiento, x_validacion, y_entrenamiento, y_validacion = train_test_split(
        x_temp, y_temp, test_size=proporcion_val_desde_temp, random_state=42, stratify=y_temp
    )

    salida_entrenamiento = x_entrenamiento.copy()
    salida_entrenamiento["estado_binario"] = y_entrenamiento.values
    salida_validacion = x_validacion.copy()
    salida_validacion["estado_binario"] = y_validacion.values
    salida_prueba = x_prueba.copy()
    salida_prueba["estado_binario"] = y_prueba.values

    salida_entrenamiento.to_csv("train.csv", index=False)
    salida_validacion.to_csv("val.csv", index=False)
    salida_prueba.to_csv("test.csv", index=False)

    escalador = StandardScaler()
    x_entrenamiento_escalado = escalador.fit_transform(x_entrenamiento)
    x_validacion_escalado = escalador.transform(x_validacion)
    x_prueba_escalado = escalador.transform(x_prueba)

    modelo_lr = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        n_jobs=None,
        solver="lbfgs",
    )

    modelo_rf = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )

    resultado_lr = evaluar_modelo(
        "RegresiónLogística",
        modelo_lr,
        (x_entrenamiento_escalado, y_entrenamiento),
        (x_validacion_escalado, y_validacion),
        x_prueba_escalado,
        y_prueba,
        prefijo_salida="lab01",
    )

    resultado_rf = evaluar_modelo(
        "RandomForest",
        modelo_rf,
        (x_entrenamiento, y_entrenamiento),
        (x_validacion, y_validacion),
        x_prueba,
        y_prueba,
        prefijo_salida="lab01",
    )

    mejor = resultado_rf if resultado_rf["auc_validacion"] >= resultado_lr["auc_validacion"] else resultado_lr
    print("\nMejor modelo según AUC de validación:", mejor["nombre"])

    total_correos = 50000
    tasa_phishing = 0.15
    tasa_legitimo = 1.0 - tasa_phishing

    matriz = mejor["matriz_prueba"]
    tn = int(matriz[0, 0])
    fp = int(matriz[0, 1])
    fn = int(matriz[1, 0])
    tp = int(matriz[1, 1])

    tasa_verdaderos_positivos = tp / (tp + fn) if (tp + fn) else 0.0
    tasa_falsos_positivos = fp / (fp + tn) if (fp + tn) else 0.0

    esperado_phishing = int(round(total_correos * tasa_phishing))
    esperado_legitimo = int(round(total_correos * tasa_legitimo))

    esperado_tp = int(round(esperado_phishing * tasa_verdaderos_positivos))
    esperado_fn = esperado_phishing - esperado_tp
    esperado_fp = int(round(esperado_legitimo * tasa_falsos_positivos))
    esperado_tn = esperado_legitimo - esperado_fp

    alarmas = esperado_tp + esperado_fp
    positivos = alarmas
    negativos = esperado_tn + esperado_fn

    print("\nEscenario de negocio (50,000 correos, 15% phishing) usando tasas del conjunto de PRUEBA del mejor modelo:")
    print("Alarmas estimadas (predicho phishing):", alarmas)
    print("Predichos phishing:", positivos)
    print("Predichos legítimos:", negativos)
    print("Estimación TP:", esperado_tp, "FP:", esperado_fp, "TN:", esperado_tn, "FN:", esperado_fn)


if __name__ == "__main__":
    main()
