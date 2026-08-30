'''Compara mediante PCA las distribuciones actualizadas de USGPV y CRSST21.'''

from src.usgpv import analisis_estadistico_pca


def main(argv=None):
    analisis_estadistico_pca()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
