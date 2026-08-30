'''Ejecuta el flujo mínimo de firma y actualización de USGPV.

El script crea el sistema, genera las claves iniciales, firma un mensaje,
verifica la firma, avanza una epoca, actualiza la firma y vuelve a verificarla.
'''

import argparse

from src import usgpv


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--message", default="mensaje de prueba")
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--cupd", type=int, default=22900)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)

    system = usgpv(args.dimension, args.epochs, args.cupd, seed=args.seed)
    system.setup()
    public_key_0, _ = system.KeyGen()
    ve_public_key_0 = system.VE_impostor_KeyGen()

    signature_0 = system.Sig(args.message)
    valid_0 = system.Ver(ve_public_key_0, public_key_0, args.message, signature_0)
    print(f"Firma inicial valida: {valid_0}")

    public_key_1, _, token = system.Next(public_key_0)
    signature_1 = system.Update(token, signature_0)
    valid_1 = system.Ver(system.VE_pk, public_key_1, args.message, signature_1)
    print(f"Firma actualizada valida: {valid_1}")

    return 0 if valid_0 and valid_1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
