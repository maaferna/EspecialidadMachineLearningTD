# src/data/images.py
from typing import Tuple, Optional, Dict
import os
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE

def _augmenter(img_size: int) -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="augmenter",
    )

def load_flowers_from_dir(
    data_dir: str = "data/flower_photos",
    img_size: int = 224,
    batch_size: int = 32,
    val_split: float = 0.2,
    seed: int = 42,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, int, Dict[int, str]]:
    """Carga flower_photos con image_dataset_from_directory y split train/val; test=val por simplicidad."""
    assert os.path.isdir(data_dir), f"No existe {data_dir}. Ejecuta bash get_flowers.sh"
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=(img_size, img_size),
        batch_size=batch_size,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=(img_size, img_size),
        batch_size=batch_size,
    )
    class_names = train_ds.class_names
    num_classes = len(class_names)
    idx_to_name = {i: n for i, n in enumerate(class_names)}

    # En este flujo: usamos val como test simplificado 
    test_ds = val_ds

    # cache/prefetch
    train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(AUTOTUNE)
    test_ds  = test_ds.cache().prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, num_classes, idx_to_name


def load_cifar10(
    img_size: int = 224,
    batch_size: int = 32,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, int, Dict[int, str]]:
    """Carga CIFAR-10 y lo reescala a img_size para TL."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    class_names = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
    num_classes = len(class_names)

    def _prep(x, y):
        x = tf.image.resize(tf.cast(x, tf.float32), (img_size, img_size))
        return x, tf.squeeze(y, axis=-1)

    ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).map(_prep, num_parallel_calls=AUTOTUNE)
    ds_test  = tf.data.Dataset.from_tensor_slices((x_test,  y_test)).map(_prep, num_parallel_calls=AUTOTUNE)

    # split val desde train
    val_split = int(0.2 * len(x_train))
    ds_val   = ds_train.take(val_split)
    ds_train = ds_train.skip(val_split)

    ds_train = ds_train.shuffle(10000).batch(batch_size).prefetch(AUTOTUNE)
    ds_val   = ds_val.batch(batch_size).prefetch(AUTOTUNE)
    ds_test  = ds_test.batch(batch_size).prefetch(AUTOTUNE)

    return ds_train, ds_val, ds_test, num_classes, {i:n for i,n in enumerate(class_names)}


def add_preprocessing_and_augmentation(
    ds: tf.data.Dataset,
    base_preprocess: tf.keras.layers.Layer,
    img_size: int,
    training: bool = True,
) -> tf.data.Dataset:
    """Aplica preprocesamiento propio del modelo + augmentación (solo en train)."""
    aug = _augmenter(img_size)
    def _map_train(x, y):
        x = aug(x)
        x = base_preprocess(x)
        return x, y

    def _map_eval(x, y):
        x = base_preprocess(x)
        return x, y

    return ds.map(_map_train if training else _map_eval, num_parallel_calls=AUTOTUNE)



def get_backbone_and_preprocess(base: str):
    """Wrapper para mantener compatibilidad con código que importa desde data.images."""
    from src.models.tl import get_backbone_and_preprocess as _get
    return _get(base)




