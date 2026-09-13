import time
import numpy as np
import argparse

from src.usgpv import usgpv


def spectral_norm(M):
    return np.linalg.norm(
        np.asarray(M, dtype=np.float64),
        ord=2
    )


def vector_norm(v):
    return np.linalg.norm(
        np.asarray(v, dtype=np.float64),
        ord=2
    )


def max_abs(v):
    arr = np.asarray(v)

    # dtype=object evita problemas al tomar abs de int64 mínimo.
    return max(abs(int(x)) for x in arr.reshape(-1))


def main(argv=None):

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--cupd", type=int, default=22900)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--message", type=str, default="prueba de escalabilidad USGPV")
    args = parser.parse_args(argv)

    print("=" * 80)
    print("PRUEBA DE ESCALABILIDAD USGPV")
    print("=" * 80)

    system = usgpv(
        args.dimension,
        args.epochs,
        args.cupd,
        seed=args.seed
    )

    system.setup()

    pk, sk = system.KeyGen()
    vepk = system.VE_impostor_KeyGen()

    sign = system.Sig(args.message)

    if not system.Ver(vepk, pk, args.message, sign):
        raise RuntimeError(
            "La firma inicial no verifica"
        )

    int64_max = np.iinfo(np.int64).max

    print()
    print(
        f"{'e':>3} "
        f"{'s_e':>16} "
        f"{'r*s_e':>16} "
        f"{'beta_e':>16} "
        f"{'||sign||_2':>16} "
        f"{'max|sign|':>16}"
    )

    print("-" * 90)

    u = sign[0]

    print(
        f"{0:>3} "
        f"{system.s_params[0]:>16.6e} "
        f"{system.r_param * system.s_params[0]:>16.6e} "
        f"{system.beta_params[0]:>16.6e} "
        f"{vector_norm(u):>16.6e} "
        f"{max_abs(u):>16.6e}"
    )

    for e in range(1, args.epochs):

        print()
        print("=" * 80)
        print(f"ACTUALIZACIÓN A ÉPOCA {e}")
        print("=" * 80)

        scale = (
            system.r_param
            * system.s_params[e]
        )

        print(
            f"s_{e}       = "
            f"{system.s_params[e]:.6e}"
        )

        print(
            f"r * s_{e}   = "
            f"{scale:.6e}"
        )

        print(
            f"int64 max   = "
            f"{int64_max:.6e}"
        )

        print(
            f"(r*s_e)/int64_max = "
            f"{scale / int64_max:.6e}"
        )

        if scale >= int64_max:
            print()
            print(
                "ADVERTENCIA: la escala gaussiana ya "
                "alcanza o supera el rango de int64."
            )

        try:

            start = time.perf_counter()

            pk_new, sk_new, token = system.Next(pk)

            next_time = (
                time.perf_counter()
                - start
            )

            token_norm = spectral_norm(token)

            start = time.perf_counter()

            sign_new = system.Update(
                token,
                sign
            )

            update_time = (
                time.perf_counter()
                - start
            )

            valid = system.Ver(
                system.VE_pk,
                pk_new,
                args.message,
                sign_new
            )

            u_new = sign_new[0]

            sign_norm = vector_norm(u_new)
            sign_max = max_abs(u_new)

            print()
            print(
                f"||Delta_{e}||_2 = "
                f"{token_norm:.6e}"
            )

            print(
                f"||sign_{e}||_2  = "
                f"{sign_norm:.6e}"
            )

            print(
                f"max|sign_{e}|   = "
                f"{sign_max:.6e}"
            )

            print(
                f"beta_{e}        = "
                f"{system.beta_params[e]:.6e}"
            )

            print(
                f"Next time        = "
                f"{next_time:.3f} s"
            )

            print(
                f"Update time      = "
                f"{update_time:.3f} s"
            )

            print(
                f"Verify           = "
                f"{valid}"
            )

            if not valid:
                print(
                    "\nSTOP: la firma actualizada "
                    "ya no verifica."
                )
                break

            print()
            print(
                f"{e:>3} "
                f"{system.s_params[e]:>16.6e} "
                f"{scale:>16.6e} "
                f"{system.beta_params[e]:>16.6e} "
                f"{sign_norm:>16.6e} "
                f"{sign_max:>16.6e}"
            )

            pk = pk_new
            sk = sk_new
            sign = sign_new

        except Exception as exc:

            print()
            print("=" * 80)
            print(
                f"FALLO EN ÉPOCA {e}"
            )
            print("=" * 80)

            print(
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            break


if __name__ == "__main__":
    raise SystemExit(
            'Ejecuta este modulo mediante uno de los lanzadores de scripts/ '
            'o mediante: python ejecutar.py --help'
        )
    