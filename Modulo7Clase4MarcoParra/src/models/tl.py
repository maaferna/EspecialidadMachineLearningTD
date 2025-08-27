# src/models/tl.py
from typing import Literal, Optional, Tuple
import tensorflow as tf
from tensorflow.keras import layers, Model

# Mapeo base -> (constructor, preprocess_input, default_input)
_BACKBONES = {
    "resnet50": (
        tf.keras.applications.ResNet50,
        tf.keras.applications.resnet50.preprocess_input,
        (224, 224, 3),
    ),
    "efficientnetb0": (
        tf.keras.applications.EfficientNetB0,
        tf.keras.applications.efficientnet.preprocess_input,
        (224, 224, 3),
    ),
}

def get_backbone_and_preprocess(
    base: Literal["resnet50", "efficientnetb0"]
):
    Constructor, preprocess, default_input = _BACKBONES[base]
    return Constructor, preprocess, default_input

def build_transfer_model(
    base: Literal["resnet50", "efficientnetb0"] = "resnet50",
    num_classes: int = 5,
    input_shape: Optional[Tuple[int,int,int]] = None,
    dropout: float = 0.2,
    train_base: bool = False,
    fine_tune_at: Optional[int] = None,
) -> Model:
    """Crea modelo TL: Backbone->GlobalPool->Dropout->Dense(num_classes)."""
    Constructor, _, default_input = get_backbone_and_preprocess(base)
    if input_shape is None:
        input_shape = default_input

    inputs = layers.Input(shape=input_shape)
    base_model = Constructor(include_top=False, weights="imagenet", input_tensor=inputs)

    base_model.trainable = train_base
    if train_base and fine_tune_at is not None:
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

    x = layers.GlobalAveragePooling2D()(base_model.output)
    if dropout and dropout > 0:
        x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return Model(inputs=inputs, outputs=outputs, name=f"tl_{base}")
