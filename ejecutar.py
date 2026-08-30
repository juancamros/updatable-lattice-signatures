'''Punto de entrada central para la implementación y los análisis del TFM.

El modulo traduce un nombre de comando a su lanzador correspondiente y lo
importa solo cuando se va a ejecutar. De este modo, pedir ayuda o ejecutar la
demostracion corta no inicia por accidente los analisis de mayor coste.
'''

import argparse
import importlib


COMMANDS = {
    "demo": "scripts.demostracion_usgpv",
    "rendimiento-usgpv": "scripts.rendimiento_usgpv",
    "rendimiento-crsst21": "scripts.rendimiento_crsst21",
    "covarianza": "scripts.analisis_covarianza",
    "pca": "scripts.analisis_pca",
    "correccion-gaussiana": "scripts.demostracion_correccion_gaussiana",
    "graficas-dispersion": "scripts.graficas_dispersion",
}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Ejecuta la implementacion o uno de los analisis del TFM."
    )
    parser.add_argument("command", choices=COMMANDS)
    args, command_args = parser.parse_known_args(argv)

    module = importlib.import_module(COMMANDS[args.command])
    return module.main(command_args)


if __name__ == "__main__":
    raise SystemExit(main())
