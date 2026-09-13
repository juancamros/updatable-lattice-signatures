import argparse

from src import usgpv
import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--cupd", type=int, default=22900)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args(argv)

    # ============================================================
    # Inicialización
    # ============================================================

    system = usgpv(
        args.dimension,
        args.epochs,
        args.cupd,
        seed=args.seed
    )

    system.setup()

    # Claves de época 0
    public_key_0, private_key_0 = system.KeyGen()

    print(f'||R|| --> {np.linalg.norm(system.R_matrix,ord=2)}')

    # Generamos las claves de época 1.
    # Ignoramos el token generado por Next(),
    # ya que generaremos nuestros propios tokens.
    public_key_1, private_key_1, _ = system.Next(public_key_0)

    # Listas para almacenar resultados
    norms_sign = []
    norms_token = []
    reductions = []

    print("=" * 70)
    print(f"GENERANDO {args.samples} PARES DE TOKENS")
    print("=" * 70)

    # ============================================================
    # Experimento
    # ============================================================

    for i in range(args.samples):

        print(f"\nPar {i + 1}/{args.samples}")

        # --------------------------------------------------------
        # Token generado con la dispersión de firma
        # --------------------------------------------------------

        token_sign = system.SamplePre(
            public_key_0,
            mode="sign"
        )

        # --------------------------------------------------------
        # Token generado con la nueva dispersión de token
        # --------------------------------------------------------

        token_token = system.SamplePre(
            public_key_0,
            mode="token"
        )

        # --------------------------------------------------------
        # Comprobación algebraica
        # --------------------------------------------------------

        valid_sign = np.array_equal(
            (public_key_1 @ token_sign) % system.q_param,
            public_key_0 % system.q_param
        )

        valid_token = np.array_equal(
            (public_key_1 @ token_token) % system.q_param,
            public_key_0 % system.q_param
        )

        if not valid_sign:
            raise ValueError(
                f"El token modo firma del par {i + 1} no es válido"
            )

        if not valid_token:
            raise ValueError(
                f"El token modo token del par {i + 1} no es válido"
            )

        # --------------------------------------------------------
        # Norma espectral
        # --------------------------------------------------------

        norm_sign = np.linalg.norm(token_sign, ord=2)
        norm_token = np.linalg.norm(token_token, ord=2)

        norms_sign.append(norm_sign)
        norms_token.append(norm_token)

        # Reducción porcentual para este par
        reduction = (
            1.0 - norm_token / norm_sign
        ) * 100

        reductions.append(reduction)

        print(
            f"  ||T_sign||₂  = {norm_sign:.4f}"
        )

        print(
            f"  ||T_token||₂ = {norm_token:.4f}"
        )

        print(
            f"  Reducción    = {reduction:.2f}%"
        )

    # ============================================================
    # Convertimos a arrays
    # ============================================================

    norms_sign = np.asarray(norms_sign)
    norms_token = np.asarray(norms_token)
    reductions = np.asarray(reductions)

    # ============================================================
    # Estadísticas
    # ============================================================

    mean_sign = np.mean(norms_sign)
    mean_token = np.mean(norms_token)

    std_sign = np.std(norms_sign, ddof=1)
    std_token = np.std(norms_token, ddof=1)

    median_sign = np.median(norms_sign)
    median_token = np.median(norms_token)

    # Reducción calculada a partir de las medias
    reduction_means = (
        1.0 - mean_token / mean_sign
    ) * 100

    # Media de las reducciones de cada pareja
    mean_pairwise_reduction = np.mean(reductions)

    # ============================================================
    # Resultados
    # ============================================================

    print("\n")
    print("=" * 70)
    print("RESULTADOS FINALES")
    print("=" * 70)

    print(f"Número de muestras: {args.samples}")

    print("\n--- MODO FIRMA ---")
    print(f"Media:              {mean_sign:.6f}")
    print(f"Mediana:            {median_sign:.6f}")
    print(f"Desviación típica:  {std_sign:.6f}")
    print(f"Mínimo:             {np.min(norms_sign):.6f}")
    print(f"Máximo:             {np.max(norms_sign):.6f}")

    print("\n--- MODO TOKEN ---")
    print(f"Media:              {mean_token:.6f}")
    print(f"Mediana:            {median_token:.6f}")
    print(f"Desviación típica:  {std_token:.6f}")
    print(f"Mínimo:             {np.min(norms_token):.6f}")
    print(f"Máximo:             {np.max(norms_token):.6f}")

    print("\n--- COMPARACIÓN ---")

    print(
        f"Token nuevo respecto al antiguo: "
        f"{(mean_token / mean_sign) * 100:.2f}%"
    )

    print(
        f"Reducción usando las medias: "
        f"{reduction_means:.2f}%"
    )

    print(
        f"Media de las reducciones por pareja: "
        f"{mean_pairwise_reduction:.2f}%"
    )

    print("=" * 70)


if __name__ == "__main__":
    raise SystemExit(main())