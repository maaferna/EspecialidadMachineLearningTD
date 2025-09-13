from pathlib import Path

OUT = Path("reports"); OUT.mkdir(parents=True, exist_ok=True)

def section(title: str) -> str:
    return f"\\n\\n## {title}\\n\\n"

def main():
    """Genera un reporte consolidado en reports/informe_interpretabilidad.md
    incluyendo métricas, gráficos SHAP y LIME, y análisis de sesgo.
    Requiere que se hayan generado previamente los gráficos y métricas.
    Args:
        None
    Returns:
        None
    """
    md = ["# Informe — Interpretabilidad con LIME & SHAP"]

    # 1. Carga y exploración (instrucciones)
    md.append(section("1) Carga y exploración de datos"))
    md.append("- Dataset: Heart Failure Prediction (Kaggle).")
    md.append("- Limpieza mínima: `dropna`, casting categóricas.")
    md.append("- Variables potencialmente sensibles: `Sex` (u otras relacionadas).")

    # 2. Modelo
    md.append(section("2) Construcción y evaluación del modelo"))
    metrics_path = Path("outputs/metrics.txt")
    if metrics_path.exists():
        md.append("**Métricas (test):**\n")
        md.append("```")
        md.append(metrics_path.read_text(encoding="utf-8"))
        md.append("```")
    else:
        md.append("_Ejecuta `python -m src.models.train` para generar métricas._")

    # 3. SHAP
    md.append(section("3) SHAP — Explicaciones"))
    md.append("Gráficos globales y locales:")
    for name in ["shap_summary_beeswarm.png", "shap_summary_bar.png", "shap_waterfall_case_1.png", "shap_waterfall_case_2.png", "shap_waterfall_case_3.png"]:
        p = Path("images")/name
        if p.exists(): md.append(f"![{name}](../images/{name})")

    eb = Path("reports/ethics_bias_shap.md")
    if eb.exists():
        md.append("\n**Análisis de posibles sesgos (SHAP):**\n")
        md.append(eb.read_text(encoding="utf-8"))

    # 4. LIME
    md.append(section("4) LIME — Explicaciones locales"))
    for name in ["lime_explanation_case_1.png", "lime_explanation_case_2.png", "lime_explanation_case_3.png"]:
        p = Path("images")/name
        if p.exists(): md.append(f"![{name}](../images/{name})")

    # 5. Sesgo y ética
    md.append(section("5) Análisis de sesgo y ética"))
    md.append("- Revisar si `Sex` (u otra variable sensible) tiene peso desproporcionado.")
    md.append("- Mitigación potencial: balanceo, eliminación o anonimización de variables, regularización, ajuste de umbral, post-procesamiento de fairness.")
    md.append("- Riesgo sin interpretabilidad: decisiones clínicas injustas o no auditables.")

    # 6. Propuesta de mejora
    md.append(section("6) Propuesta de mejora"))
    md.append("- Evaluar otro algoritmo (e.g., XGBoost) y comparar métricas + explicaciones.")
    md.append("- Ajustar preprocesamiento (codificación, estandarización, manejo de outliers).")
    md.append("- Re-entrenar sin variables sensibles y comparar desempeño/explicaciones.")

    # Guardar
    out_path = OUT/"informe_interpretabilidad.md"
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"Reporte consolidado: {out_path}")

if __name__ == "__main__":
    main()
