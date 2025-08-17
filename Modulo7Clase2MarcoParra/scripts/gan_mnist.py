# scripts/gan_mnist.py
import os, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.gan import build_generator, build_discriminator
from src.experiments.gan_trainer import train_gan_mnist

def main(out_dir: str = "outputs/gan",
         iterations: int = 3000,
         batch_size: int = 128,
         noise_dim: int = 100,
         sample_every: int = 500):
    os.makedirs(out_dir, exist_ok=True)

    G = build_generator(noise_dim=noise_dim)
    D = build_discriminator()

    print(G.summary())
    print(D.summary())

    hist = train_gan_mnist(
        generator=G,
        discriminator=D,
        out_dir=out_dir,
        iterations=iterations,
        batch_size=batch_size,
        noise_dim=noise_dim,
        sample_every=sample_every
    )
    print("✅ GAN entrenamiento finalizado.")
    print(f"Se guardaron muestras cada {sample_every} iteraciones en: {out_dir}")

if __name__ == "__main__":
    main()
