# ------------------------------ utilidades -----------------------------------
import os
import json
import random
from typing import Any, Dict
import numpy as np
import tensorflow as tf

def _to_native(obj):
    """Convierte tensores/np.* a tipos nativos para JSON."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def save_history_json(history, out_path: str):
    payload = {k: [_to_native(v) for v in vals] for k, vals in history.history.items()}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)





def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_gpus(use_mixed: bool = False, use_distributed: bool = True):
    """Configura memoria, mixed precision y estrategia de distribución."""
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

    if use_mixed:
        try:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy("mixed_float16")
            print("Mixed precision ON")
        except Exception as e:
            print("No se pudo activar mixed precision:", e)

    if use_distributed and len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
        print(f"Estrategia: MirroredStrategy | GPUs visibles: {len(gpus)}")
        return strategy

    print(f"Estrategia: Default (1 dispositivo) | GPUs visibles: {len(gpus)}")
    return tf.distribute.get_strategy()


def _default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    raise TypeError(f"Tipo no serializable: {type(obj)}")


def safe_save_json(d: Dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(d, f, indent=2, default=_default)



import os
import random
import numpy as np
import tensorflow as tf

def set_seed(seed: int = 42) -> None:
  """Fija semillas para reproducibilidad."""
  os.environ["PYTHONHASHSEED"] = str(seed)
  random.seed(seed)
  np.random.seed(seed)
  tf.random.set_seed(seed)

def enable_memory_growth() -> list[tf.config.PhysicalDevice]:
  """Activa memory growth en todas las GPUs visibles."""
  gpus = tf.config.list_physical_devices("GPU")
  for g in gpus:
    try:
      tf.config.experimental.set_memory_growth(g, True)
    except Exception:
      pass
  return gpus

def setup_strategy(prefer_mirrored: bool = True) -> tf.distribute.Strategy:
  """
  Devuelve una estrategia de distribución:
  - MirroredStrategy si hay >=1 GPU y prefer_mirrored=True.
  - OneDeviceStrategy (CPU) en caso contrario.
  """
  gpus = tf.config.list_physical_devices("GPU")
  if prefer_mirrored and gpus:
    return tf.distribute.MirroredStrategy()
  return tf.distribute.OneDeviceStrategy(device="/cpu:0")

