# Arquitectura del código

Este documento explica cómo se organiza la prueba de concepto y qué ocurre
cuando se ejecuta cada parte. El código tiene dos niveles claramente separados:

- `src/` contiene las implementaciones y las funciones matemáticas.
- `scripts/` contiene los programas que ejecutan demostraciones y experimentos.

`ejecutar.py` es el punto de entrada central y permite seleccionar uno de esos
programas sin importar ni ejecutar innecesariamente los demás.

## Flujo general

```text
ejecutar.py
    |
    +-- scripts/demostracion_usgpv.py
    |       |
    |       +-- src/usgpv.py
    |               |
    |               +-- src/ve_impostor.py
    |
    +-- scripts/rendimiento_usgpv.py ------> src/usgpv.py
    +-- scripts/rendimiento_crsst21.py ----> src/crsst21.py
    +-- scripts/analisis_covarianza.py ----> src/usgpv.py
    +-- scripts/analisis_pca.py
    |       +-- src/usgpv.py
    |       +-- src/crsst21.py
    |
    +-- scripts/demostracion_correccion_gaussiana.py
    |       +-- src/correccion_gaussiana.py
    |
    +-- scripts/graficas_dispersion.py
            +-- crecimiento_dispersion.py
```

## Implementación USGPV

`src/usgpv.py` contiene la clase `usgpv`, que representa un sistema para un
usuario y mantiene el estado de las distintas épocas.

Su flujo básico es:

1. `setup()` calcula los parámetros derivados: módulo, dimensiones, parámetros
   gaussianos y cotas de aceptación.
2. `KeyGen()` genera la clave pública y la trapdoor de la época inicial.
3. `VE_impostor_KeyGen()` crea las claves usadas para encapsular el testigo.
4. `Sig(mensaje)` calcula una preimagen corta y construye la firma.
5. `Ver(...)` comprueba la ecuación de verificación y la cota de la firma.
6. `Next(clave_publica)` genera la siguiente clave y el token de actualización.
7. `Update(token, firma)` transforma la firma, añade el ruido corrector y
   encapsula el nuevo testigo.

La instancia conserva registros de claves, tokens y firmas por época. Esto
facilita los experimentos, pero también implica que es un prototipo con estado,
no una API criptográfica preparada para producción.

## Implementación CRSST21

`src/crsst21.py` contiene la clase `crsst21`. Sigue una interfaz similar a
USGPV para que ambos esquemas puedan ejecutarse con parámetros comparables.

Su finalidad principal es actuar como línea base experimental. Permite comparar
el tiempo, la memoria y la distribución de las firmas actualizadas. No es una
dependencia conceptual del algoritmo USGPV; se utiliza en las comparaciones del
TFM y en el análisis PCA.

## Componente VE impostor

`src/ve_impostor.py` serializa matrices NumPy y las cifra con `SealedBox` de
PyNaCl.

Las funciones principales son:

- `VE_keygen()`: genera una clave pública y otra privada.
- `vector_to_bytes()`: serializa un vector sin utilizar `pickle`.
- `bytes_to_vector()`: reconstruye el vector serializado.
- `VE_impostor_encrypt()`: cifra el testigo con la clave pública.
- `VE_impostor_decrypt()`: recupera el testigo con la clave privada.

Este módulo no implementa un esquema completo de cifrado verificable. Es un
sustituto funcional que permite evaluar el flujo del prototipo.

## Corrección gaussiana

`src/correccion_gaussiana.py` ilustra en dos dimensiones la operación estadística
utilizada durante una actualización.

Parte de una muestra con covarianza inicial, aplica una transformación lineal y
calcula la covarianza del ruido que debe añadirse para aproximar la covarianza
objetivo. También comprueba que esa matriz sea semidefinida positiva antes de
muestrear.

El módulo ayuda a comprender la idea matemática, pero no interviene directamente
en las clases `usgpv` o `crsst21`.

## Crecimiento de la dispersión

`crecimiento_dispersion.py` implementa las expresiones utilizadas para estimar
el crecimiento de `s_e`.

- `c_upd(p0)` calcula la estimación del factor de actualización.
- `s_n(p0, epoch)` calcula la dispersión estimada en una época.
- `plot_by_probability()` compara varias épocas al variar `p0`.
- `plot_by_epoch()` muestra el crecimiento entre épocas para un `p0` fijo.

Las gráficas solo se muestran al llamar a `main()`. Importar el archivo no tiene
efectos visuales.

## Lanzadores

Los archivos de `scripts/` contienen poco código deliberadamente. Su función es
escoger una operación concreta, importar únicamente lo necesario y ejecutarla.
Esta separación evita que la implementación matemática dependa de la interfaz
de línea de comandos.

El selector `ejecutar.py` importa dinámicamente el lanzador elegido. Por ejemplo:

```powershell
python ejecutar.py demo --message "hola"
python ejecutar.py covarianza
python ejecutar.py pca
```

Los análisis de rendimiento, covarianza y PCA pueden ser costosos. La
demostración `demo` es el punto de partida recomendado para comprobar el flujo
funcional.

## Limitaciones

- Es una prueba de concepto académica, no código criptográfico de producción.
- El componente VE es un sustituto funcional.
- El muestreo de retículos es una aproximación experimental.
- El crecimiento rápido de los parámetros puede causar overflow.
- Las semillas fijas sirven para reproducibilidad, no para seguridad.
- Las clases guardan claves secretas y firmas en memoria para facilitar los
  experimentos.
