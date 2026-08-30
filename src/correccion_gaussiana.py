'''Demostración bidimensional del ruido gaussiano de corrección.

Este modulo aisla la idea estadistica utilizada durante una actualizacion:
primero transforma una muestra gaussiana con ``Delta`` y despues suma un ruido
cuya covarianza compensa la deformacion introducida por esa transformacion.
'''

import numpy as np
import matplotlib.pyplot as plt


def sample_update_demo(
    n_samples=5000,
    s_e=3.0,
    s_next=3.0,
    Delta=None,
    seed=123,
    round_samples=True,
):
    rng = np.random.default_rng(seed)

    if Delta is None:
        Delta = np.array([
            [1.4, 0.6],
            [0.2, 1.1]
        ], dtype=float)
    else:
        Delta = np.asarray(Delta, dtype=float)

    if Delta.shape != (2, 2):
        raise ValueError('Delta debe ser una matriz 2x2.')

    I = np.eye(2)

    # Covarianza inicial: Sigma_e = s_e^2 I
    Sigma_e = 3 * I

    # Covarianza del término transformado: Delta Sigma_e Delta^T
    transformed_cov = (Delta @ Delta.T)

    # Covarianza del ruido corrector:
    # Sigma_upd = s_{e+1}^2 I - s_e^2 Delta Delta^T
    Sigma_upd = 6 * I - transformed_cov

    # Simetrizamos por seguridad numérica
    Sigma_upd = (Sigma_upd + Sigma_upd.T) / 2

    # Comprobamos que la matriz sea semidefinida positiva
    eigvals = np.linalg.eigvalsh(Sigma_upd)
    if eigvals.min() < -1e-10:
        Delta_norm = np.linalg.norm(Delta, 2)
        s_next_min = s_e * Delta_norm

        raise ValueError(
            'Sigma_upd no es semidefinida positiva.\n'
            f'Autovalores de Sigma_upd: {eigvals}\n'
            f'Condición necesaria: s_next >= s_e * ||Delta||_2\n'
            f'||Delta||_2 = {Delta_norm:.4f}\n'
            f's_next mínimo = {s_next_min:.4f}\n'
            f's_next actual = {s_next:.4f}'
        )

    # Eliminamos posibles valores negativos minúsculos debidos al redondeo
    Sigma_upd = np.where(np.abs(Sigma_upd) < 1e-14, 0, Sigma_upd)

    # 1. Muestreo inicial
    X_e = rng.multivariate_normal(
        mean=np.zeros(2),
        cov=Sigma_e,
        size=n_samples
    )

    # 2. Transformación lineal
    X_transformed = X_e @ Delta.T

    # 3. Ruido corrector
    R_corr = rng.multivariate_normal(
        mean=np.zeros(2),
        cov=Sigma_upd,
        size=n_samples
    )

    # Redondeo opcional a enteros para simular el paso discreto
    if round_samples:
        X_e = np.rint(X_e).astype(int)
        X_transformed = np.rint(X_transformed).astype(int)
        R_corr = np.rint(R_corr).astype(int)

    # 4. Resultado actualizado
    X_next = X_transformed + R_corr

    print('Delta =')
    print(Delta)

    print('\n||Delta||_2 =', np.linalg.norm(Delta, 2))

    print('\nSigma_e =')
    print(Sigma_e)

    print('\nDelta Sigma_e Delta^T =')
    print(transformed_cov)

    print('\nSigma_upd =')
    print(Sigma_upd)

    print('\nAutovalores de Sigma_upd =')
    print(eigvals)

    print('\nCovarianza empírica de X_e:')
    print(np.cov(X_e.T))

    print('\nCovarianza empírica de Delta X_e:')
    print(np.cov(X_transformed.T))

    print('\nCovarianza empírica del ruido corrector:')
    print(np.cov(R_corr.T))

    print('\nCovarianza empírica de X_next:')
    print(np.cov(X_next.T))

    print('\nCovarianza objetivo s_next^2 I:')
    print((s_next ** 2) * I)

    # Gráficas
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].scatter(X_e[:, 0], X_e[:, 1], s=4, alpha=0.3)
    axes[0].set_title(r'$X \sim \mathcal{N}(\mu,\Sigma)$')
    axes[0].axis('equal')
    axes[0].grid(True)

    axes[1].scatter(X_transformed[:, 0], X_transformed[:, 1], s=4, alpha=0.3)
    axes[1].set_title(r'$Y = A X$')
    axes[1].axis('equal')
    axes[1].grid(True)

    axes[2].scatter(X_next[:, 0], X_next[:, 1], s=4, alpha=0.3)
    axes[2].set_title(r'$Y=A X+r$')
    axes[2].axis('equal')
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

    return X_e, X_transformed, R_corr, X_next, Sigma_upd


if __name__ == '__main__':
    raise SystemExit(
        'Ejecuta esta demostración con: '
        'python -m scripts.demostracion_correccion_gaussiana'
    )
