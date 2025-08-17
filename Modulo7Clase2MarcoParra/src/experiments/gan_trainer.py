import os
import numpy as np
import tensorflow as tf
from typing import Dict
from src.visualizer.gan_plots import save_image_grid, plot_gan_losses

# Configuración de tf.data para optimización
AUTOTUNE = tf.data.AUTOTUNE

def make_mnist_dataset(batch_size: int = 128):
    (x_train, _), _ = tf.keras.datasets.mnist.load_data()
    x_train = (x_train.astype("float32") - 127.5) / 127.5  # -> [-1,1]
    x_train = np.expand_dims(x_train, axis=-1)             # (N,28,28,1)
    ds = tf.data.Dataset.from_tensor_slices(x_train).shuffle(60000).batch(batch_size).prefetch(AUTOTUNE)
    return ds

@tf.function
def _train_step(
    generator, discriminator, g_opt, d_opt, noise_dim, real_batch
):
    # real labels=1, fake labels=0
    batch_size = tf.shape(real_batch)[0]
    noise = tf.random.normal([batch_size, noise_dim])
    fake_images = generator(noise, training=True)

    # --- Train Discriminator ---
    with tf.GradientTape() as tape_d:
        logits_real = discriminator(real_batch, training=True)
        logits_fake = discriminator(fake_images, training=True)
        d_loss_real = tf.keras.losses.binary_crossentropy(tf.ones_like(logits_real), logits_real)
        d_loss_fake = tf.keras.losses.binary_crossentropy(tf.zeros_like(logits_fake), logits_fake)
        d_loss = tf.reduce_mean(d_loss_real + d_loss_fake)
    grads_d = tape_d.gradient(d_loss, discriminator.trainable_variables)
    d_opt.apply_gradients(zip(grads_d, discriminator.trainable_variables))

    # --- Train Generator ---
    noise = tf.random.normal([batch_size, noise_dim])
    with tf.GradientTape() as tape_g:
        fake_images = generator(noise, training=True)
        logits_fake = discriminator(fake_images, training=True)
        g_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(logits_fake), logits_fake))
    grads_g = tape_g.gradient(g_loss, generator.trainable_variables)
    g_opt.apply_gradients(zip(grads_g, generator.trainable_variables))

    return d_loss, g_loss

def train_gan_mnist(
    generator,
    discriminator,
    out_dir: str = "outputs/gan",
    iterations: int = 3000,
    batch_size: int = 128,
    noise_dim: int = 100,
    sample_every: int = 500,
) -> Dict[str, list]:
    os.makedirs(out_dir, exist_ok=True)
    ds = make_mnist_dataset(batch_size=batch_size)

    # fresh optimizers por compatibilidad Keras 3
    try:
        from tensorflow.keras.optimizers.legacy import Adam as LegacyAdam
        g_opt = LegacyAdam(learning_rate=2e-4, beta_1=0.5)
        d_opt = LegacyAdam(learning_rate=2e-4, beta_1=0.5)
    except Exception:
        g_opt = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
        d_opt = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)

    it = iter(ds)
    history = {"d_loss": [], "g_loss": []}
    fixed_noise = tf.random.normal([25, noise_dim])  # para snapshots consistentes

    for step in range(1, iterations+1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(ds)
            batch = next(it)

        d_loss, g_loss = _train_step(generator, discriminator, g_opt, d_opt, noise_dim, batch)
        history["d_loss"].append(float(d_loss.numpy()))
        history["g_loss"].append(float(g_loss.numpy()))

        if step % sample_every == 0 or step == 1:
            # muestras fijas para ver progreso
            samples = generator(fixed_noise, training=False).numpy()
            save_image_grid(samples, os.path.join(out_dir, f"samples_step_{step}.png"), nrows=5, ncols=5)
            print(f"[{step}/{iterations}] d_loss={history['d_loss'][-1]:.4f} g_loss={history['g_loss'][-1]:.4f}")

    # curva de pérdidas
    plot_gan_losses(history, os.path.join(out_dir, "gan_losses.png"))
    # modelos
    generator.save(os.path.join(out_dir, "generator.keras"))
    discriminator.save(os.path.join(out_dir, "discriminator.keras"))
    return history
