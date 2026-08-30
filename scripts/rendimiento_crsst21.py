'''Mide tiempo, CPU y memoria para la implementación comparativa CRSST21.'''

from src.crsst21 import medir_rendimiento, simulacion


def main(argv=None):
    medir_rendimiento(simulacion)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
