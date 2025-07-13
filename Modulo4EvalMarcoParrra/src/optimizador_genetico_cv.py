import random
import numpy as np
import time
from deap import base, creator, tools
from sklearn.model_selection import cross_val_score
from src.modelos import crear_modelo_random_forest
from src.evaluador import evaluar_modelo_cv

# Reproducibilidad
random.seed(42)
np.random.seed(42)

# Crear tipos si no existen
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)

def evaluar_individuo(individuo, X, y):
    """
    Evalúa un individuo con validación cruzada usando F1 macro.
    """
    n_estimators, max_depth, min_samples_split = individuo
    modelo = crear_modelo_random_forest(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split
    )
    scores = cross_val_score(modelo, X, y, cv=3, scoring="f1_macro", n_jobs=-1)
    return (scores.mean(),)

def optimizar_con_genetico_cv(X_train, y_train, n_generaciones, tuned_params):
    """
    Optimización de hiperparámetros con Algoritmo Genético usando CV.
    """
    if tuned_params is None:
        raise ValueError("tuned_params no puede ser None.")

    print("\n🧬 Optimizando con Algoritmo Genético...")

    pop_size = 40  # Población base

    toolbox = base.Toolbox()
    toolbox.register("n_estimators", random.choice, tuned_params["n_estimators"])
    toolbox.register("max_depth", random.choice, tuned_params["max_depth"])
    toolbox.register("min_samples_split", random.choice, tuned_params["min_samples_split"])

    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.n_estimators, toolbox.max_depth, toolbox.min_samples_split), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluar_individuo, X=X_train, y=y_train)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.3)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    print(f"📊 Evolución: {n_generaciones} generaciones con población de {pop_size}")
    start = time.time()

    for gen in range(n_generaciones):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.7:
                toolbox.mate(child1, child2)
                del child1.fitness.values, child2.fitness.values

        for mutant in offspring:
            if random.random() < 0.3:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalids = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalids:
            ind.fitness.values = toolbox.evaluate(ind)

        population[:] = offspring
        hof.update(population)

        record = stats.compile(population)
        print(f"📈 Generación {gen + 1} | F1 promedio: {record['avg']:.4f} | F1 máx: {record['max']:.4f}")

    end = time.time()

    best_params = hof[0]
    print("\n🏆 Mejores hiperparámetros encontrados:")
    print(f"   - n_estimators: {best_params[0]}")
    print(f"   - max_depth: {best_params[1]}")
    print(f"   - min_samples_split: {best_params[2]}")

    best_model = crear_modelo_random_forest(
        n_estimators=best_params[0],
        max_depth=best_params[1],
        min_samples_split=best_params[2]
    )
    best_model.fit(X_train, y_train)

    return {
        "metodo": "Genético",
        "mejores_parametros": {
            "n_estimators": int(hof[0][0]),
            "max_depth": int(hof[0][1]),
            "min_samples_split": int(hof[0][2]),
        },
        "modelo": best_model,
        "tiempo": end - start,
    }

