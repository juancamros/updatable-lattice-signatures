'''Mide tiempo, CPU y memoria durante la simulación de USGPV.'''

from src.usgpv import medir_rendimiento, simulacion


def main(argv=None):
    medir_rendimiento(simulacion)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
