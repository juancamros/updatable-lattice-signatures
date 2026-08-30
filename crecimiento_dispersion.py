'''Cálculo y representación del crecimiento del parámetro de dispersión.

Las funciones implementan las expresiones empleadas en el TFM para estudiar
como evoluciona ``s_e`` en funcion de ``p0`` y del numero de actualizaciones.
Importar el modulo no genera ninguna grafica; las figuras solo se muestran al
llamar a :func:`main`.
'''

import matplotlib.pyplot as plt
import numpy as np


W = 224
M = 448


def c_upd(p0):
    '''Calcula la estimación del crecimiento para una probabilidad ``p0``.'''
    return 2 * p0 * W * np.sqrt(M) * np.log2(W)


def s_n(p0, epoch):
    '''Calcula la dispersión estimada para una época determinada.'''
    growth = c_upd(p0)
    s0 = p0 * W
    return (growth**epoch) * s0 + growth * (growth**epoch - 1) / (growth - 1)


def plot_by_probability():
    '''Representa la dispersión según ``p0`` para las épocas de cero a cuatro.'''
    p0_values = np.linspace(0.01, 0.30, 300)
    epochs = [0, 1, 2, 3, 4]
    colors = plt.cm.Blues(np.linspace(0.35, 0.9, len(epochs)))

    figure, axis = plt.subplots(figsize=(8, 5))
    for color, epoch in zip(colors, epochs):
        axis.plot(
            p0_values,
            s_n(p0_values, epoch),
            label=fr"$s_{epoch}$",
            color=color,
        )

    axis.set_yscale("log")
    axis.set_xlabel("$p_0$")
    axis.set_ylabel("$s_e$")
    axis.legend()
    axis.grid(True, which="both", linestyle="--", linewidth=0.5)
    figure.tight_layout()
    return figure


def plot_by_epoch(p0=0.30):
    '''Representa la dispersión por época para un valor concreto de ``p0``.'''
    epochs = np.arange(0, 5)
    values = [s_n(p0, epoch) for epoch in epochs]

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(epochs, values, marker="o")
    axis.set_yscale("log")
    axis.set_xlabel("Epoca $e$")
    axis.set_ylabel("$s_e$")
    axis.grid(True, which="both", linestyle="--", linewidth=0.5)
    figure.tight_layout()

    for epoch, value in zip(epochs, values):
        print(f"s_{epoch} = {value:.4e}")

    return figure


def main():
    plot_by_probability()
    plot_by_epoch()
    plt.show()


if __name__ == "__main__":
    main()
