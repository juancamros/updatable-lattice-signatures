import argparse

from src import usgpv
import numpy as np

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
    public_key_0, private_key_0 = system.KeyGen()
    ve_public_key_0 = system.VE_impostor_KeyGen()

    public_key_1, private_key_1, token_1 = system.Next()

    token_red, P, U = system.LLL_token_reduction(token=token_1)

    U_inv = np.linalg.inv(U)

    new_public_key_1 = public_key_1 @ U_inv
    new_B_matrix = U @ system.B_matrix


    print('='*50)
    print('Análisis de la variación del tamaño del token y la trapdoor')
    print('='*50)

    print('-'*50)
    print('TOKEN:')
    print(f'Norma espectral del token antiguo:{np.linalg.norm(token_1,ord=2)}')
    print(f'Norma frobenius del token antiguo:{np.linalg.norm(token_1,ord='fro')}')
    print(f'Norma espectral del token nuevo:{np.linalg.norm(token_red,ord=2)}')
    print(f'Norma frobenius del token nuevo:{np.linalg.norm(token_red,ord='fro')}')
    print(f'Tasa de variación norma espectral nuevo vs antiguo:{(np.linalg.norm(token_red,ord=2)/np.linalg.norm(token_1,ord=2))*100}')
    print(f'Tasa de variación norma frobenius nuevo vs antiguo:{(np.linalg.norm(token_red,ord='fro')/np.linalg.norm(token_1,ord='fro'))*100}')
    print('-'*50)

    print('-'*50)
    print('TRAPDOOR:')
    print(f'Norma espectral de la trapdoor antigua:{np.linalg.norm(system.B_matrix,ord=2)}')
    print(f'Norma frobenius de la trapdoor antigua:{np.linalg.norm(system.B_matrix,ord='fro')}')
    print(f'Norma espectral de la trapdoor nueva:{np.linalg.norm(new_B_matrix,ord=2)}')
    print(f'Norma frobenius de la trapdoor nueva:{np.linalg.norm(new_B_matrix,ord='fro')}')
    print(f'Tasa de variación norma espectral nuevo vs antiguo:{(np.linalg.norm(new_B_matrix,ord=2)/np.linalg.norm(system.B_matrix,ord=2))*100}')
    print(f'Tasa de variación norma frobenius nuevo vs antiguo:{(np.linalg.norm(new_B_matrix,ord='fro')/np.linalg.norm(system.B_matrix,ord='fro'))*100}')
    print('-'*50)



if __name__ == "__main__":
    raise SystemExit(main())