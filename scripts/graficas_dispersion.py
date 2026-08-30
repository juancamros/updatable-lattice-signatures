'''Genera las gráficas de crecimiento del parámetro de dispersión.'''

from crecimiento_dispersion import main as plot_parameter_growth


def main(argv=None):
    plot_parameter_growth()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
