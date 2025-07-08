from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score



def evaluar_modelo(nombre, modelo, X_test, y_test, tiempo, mejores_parametros):
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n📊 Evaluación {nombre}:")
    print(f"F1-Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"⏱️ Tiempo de optimización: {tiempo:.2f} segundos")

    return {
        "metodo": nombre,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "auc": auc,
        "modelo": modelo,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "tiempo": tiempo,
        "mejores_parametros": mejores_parametros,
    }
