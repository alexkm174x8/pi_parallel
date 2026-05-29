/**
 * bmp_filters_extended.h
 * - Filtros extendidos de procesamiento de imágenes BMP.
 *
 * Complementa bmp_filters_core.h con tres transformaciones adicionales:
 *
 *   - inv_img_grey_horizontal()    Espejo horizontal + escala de grises  (filtro "hg")
 *   - inv_img_color_horizontal()   Espejo horizontal conservando colores  (filtro "hc")
 *   - desenfoque()                 Box blur en dos pasadas en escala de grises (filtro "dg")
 *
 * También incluye dos funciones de utilería internas:
 *   - manejar_encabezado_1()  Duplicado de manejar_encabezado() de core (evita dependencia cruzada).
 *   - gray_img()              Conversión a gris sin invertir (declarada pero no usada en el flujo principal).
 *
 * Todas las funciones son `static` y se incluyen en bmp_processor.c mediante #include.
 * Se invocan dentro de bloques `#pragma omp section`, corriendo en hilos OpenMP
 * concurrentes con las funciones de bmp_filters_core.h.
 *
 * Diferencia clave con bmp_filters_core.h:
 *   - Core:     inversión VERTICAL  (las filas se escriben de abajo hacia arriba).
 *   - Extended: inversión HORIZONTAL (las columnas de cada fila se escriben de derecha a izquierda).
 *
 * Formato BMP asumido: 24 bpp, orden de bytes B-G-R por píxel, padding a 4 bytes por fila.
 *
 * Dependencias: <stdio.h>, <stdlib.h>, <string.h>
 */

#ifndef BMP_FILTERS_EXTENDED_H
#define BMP_FILTERS_EXTENDED_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


/* ===========================================================================
 * Utilidades internas
 * =========================================================================*/

/**
 * Lee y copia el encabezado BMP completo de entrada a salida.
 *
 * Función idéntica a manejar_encabezado() de bmp_filters_core.h.
 * Se duplica aquí con el nombre manejar_encabezado_1 para que este header
 * sea autónomo y no dependa de la inclusión previa de bmp_filters_core.h,
 * evitando conflictos de redefinición de funciones static.
 *
 * in      Archivo BMP de entrada abierto en modo "rb".
 * out     Archivo BMP de salida abierto en modo "wb".
 * ancho   [out] Ancho de la imagen en píxeles.
 * alto    [out] Alto de la imagen en píxeles.
 * offset  [out] Desplazamiento en bytes al inicio de los datos de píxeles.
 */
static void manejar_encabezado_1(FILE *in, FILE *out, int *ancho, int *alto, int *offset) {
    unsigned char fileHeader[14];
    fread(fileHeader, 1, 14, in);
    *offset = *(int *)&fileHeader[10];

    unsigned char *fullHeader = (unsigned char *)malloc(*offset);
    memcpy(fullHeader, fileHeader, 14);
    fread(fullHeader + 14, 1, *offset - 14, in);
    fwrite(fullHeader, 1, *offset, out);

    *ancho = *(int *)&fullHeader[18];
    *alto  = *(int *)&fullHeader[22];
    free(fullHeader);
}

/**
 * Convierte la imagen a escala de grises sin ninguna transformación geométrica.
 *
 *  Esta función está declarada pero NO es invocada por ningún filtro del flujo
 *       principal (bmp_processor.c no la llama). Fue implementada como utilería
 *       durante el desarrollo y se conserva para posibles usos futuros.
 *
 * Fórmula de conversión (luminancia perceptual):
 *   gray = 0.21 * R + 0.72 * G + 0.07 * B
 *
 *  output_path  Ruta del archivo BMP de salida.
 * input_path   Ruta del archivo BMP de entrada.
 */
static void gray_img(const char *output_path, const char *input_path) {
    FILE *image = fopen(input_path, "rb");
    if (!image) {
        return;
    }

    FILE *outputImage = fopen(output_path, "wb");
    if (!outputImage) {
        fclose(image);
        return;
    }

    int ancho, alto, offset;
    manejar_encabezado_1(image, outputImage, &ancho, &alto, &offset);

    int padding = (4 - (ancho * 3) % 4) % 4;

    for (int i = 0; i < alto; i++) {
        for (int j = 0; j < ancho; j++) {
            unsigned char b = fgetc(image);
            unsigned char g = fgetc(image);
            unsigned char r = fgetc(image);
            unsigned char gray = (unsigned char)(0.21 * r + 0.72 * g + 0.07 * b);
            fputc(gray, outputImage);
            fputc(gray, outputImage);
            fputc(gray, outputImage);
        }
        for (int p = 0; p < padding; p++) {
            fgetc(image);
            fputc(0, outputImage);
        }
    }

    fclose(image);
    fclose(outputImage);
}


/* ===========================================================================
 * Filtros de imagen
 * =========================================================================*/

/**
 * Espejo horizontal con conversión a escala de grises  (filtro "hg").
 *
 * Refleja la imagen sobre el eje vertical central (espejo izquierda-derecha)
 * y convierte cada píxel a gris con la fórmula de luminancia perceptual:
 *
 *   gray = 0.21 * R + 0.72 * G + 0.07 * B
 *
 * Diferencia con inv_img() de core:
 *   - inv_img()  →  invierte el ORDEN DE LAS FILAS  (la fila 0 pasa a ser la última).
 *   - inv_img_grey_horizontal()  →  invierte el ORDEN DE LAS COLUMNAS dentro de cada fila
 *                                   (el píxel de la columna 0 pasa a la columna ancho-1).
 *
 * Algoritmo:
 *   1. Lee todos los píxeles convirtiendo a gris, almacena en arreglo lineal
 *      pixels[fila * ancho + columna].
 *   2. Escribe cada fila recorriéndola de columna (ancho-1) a 0.
 *
 * Invocado desde bmp_processor.c dentro de:
 *   #pragma omp section  { if (filters.hg) inv_img_grey_horizontal(out_hg, image_paths[i]); }
 *
 * output_path  Ruta del archivo BMP de salida. Nombre: <original>_hg.bmp
 * input_path   Ruta del archivo BMP de entrada.
 */
static void inv_img_grey_horizontal(const char *output_path, const char *input_path) {
    FILE *image = fopen(input_path, "rb");
    if (!image) {
        return;
    }

    FILE *outputImage = fopen(output_path, "wb");
    if (!outputImage) {
        fclose(image);
        return;
    }

    int ancho, alto, offset;
    manejar_encabezado_1(image, outputImage, &ancho, &alto, &offset);

    int padding = (4 - (ancho * 3) % 4) % 4;

    /* Arreglo lineal: un byte de gris por píxel */
    unsigned char *pixels = (unsigned char *)malloc(ancho * alto);

    /* Lectura: convierte a gris, almacena en orden natural (fila por fila) */
    for (int i = 0; i < alto; i++) {
        for (int j = 0; j < ancho; j++) {
            unsigned char b = fgetc(image);
            unsigned char g = fgetc(image);
            unsigned char r = fgetc(image);
            pixels[i * ancho + j] = (unsigned char)(0.21 * r + 0.72 * g + 0.07 * b);
        }
        for (int p = 0; p < padding; p++) fgetc(image);
    }

    /* Escritura: mismas filas, pero columnas en orden inverso (espejo horizontal) */
    for (int i = 0; i < alto; i++) {
        for (int j = ancho - 1; j >= 0; j--) {
            unsigned char pxl = pixels[i * ancho + j];
            /* Escribe el mismo valor en B, G, R para mantener 24 bpp en gris */
            fputc(pxl, outputImage);
            fputc(pxl, outputImage);
            fputc(pxl, outputImage);
        }
        for (int p = 0; p < padding; p++) fputc(0, outputImage);
    }

    free(pixels);
    fclose(image);
    fclose(outputImage);
}

/**
 * Espejo horizontal conservando el color original  (filtro "hc").
 *
 * Refleja la imagen sobre el eje vertical central igual que
 * inv_img_grey_horizontal(), pero sin convertir a gris: preserva
 * los canales B, G, R originales de cada píxel.
 *
 * Diferencia con inv_img_color() de core:
 *   - inv_img_color()             →  invierte FILAS,    conserva colores.
 *   - inv_img_color_horizontal()  →  invierte COLUMNAS, conserva colores.
 *
 * Algoritmo:
 *   1. Lee todos los píxeles almacenando B, G, R en tres arreglos lineales separados.
 *   2. Escribe cada fila recorriendo las columnas de (ancho-1) a 0, emitiendo
 *      los tres canales en orden B, G, R.
 *
 * Invocado desde bmp_processor.c dentro de:
 *   #pragma omp section  { if (filters.hc) inv_img_color_horizontal(out_hc, image_paths[i]); }
 *
 * output_path  Ruta del archivo BMP de salida. Nombre: <original>_hc.bmp
 * input_path   Ruta del archivo BMP de entrada.
 */
static void inv_img_color_horizontal(const char *output_path, const char *input_path) {
    FILE *image = fopen(input_path, "rb");
    if (!image) {
        return;
    }

    FILE *outputImage = fopen(output_path, "wb");
    if (!outputImage) {
        fclose(image);
        return;
    }

    int ancho, alto, offset;
    manejar_encabezado_1(image, outputImage, &ancho, &alto, &offset);

    int padding = (4 - (ancho * 3) % 4) % 4;

    /* Tres arreglos lineales, uno por canal de color */
    unsigned char *b = (unsigned char *)malloc(ancho * alto);
    unsigned char *g = (unsigned char *)malloc(ancho * alto);
    unsigned char *r = (unsigned char *)malloc(ancho * alto);

    /* Lectura: almacena cada canal en su arreglo correspondiente */
    for (int i = 0; i < alto; i++) {
        for (int j = 0; j < ancho; j++) {
            b[i * ancho + j] = fgetc(image);  /* orden BMP: B primero */
            g[i * ancho + j] = fgetc(image);
            r[i * ancho + j] = fgetc(image);
        }
        for (int p = 0; p < padding; p++) fgetc(image);
    }

    /* Escritura: columnas en orden inverso, color intacto */
    for (int i = 0; i < alto; i++) {
        for (int j = ancho - 1; j >= 0; j--) {
            int idx = i * ancho + j;
            fputc(b[idx], outputImage);
            fputc(g[idx], outputImage);
            fputc(r[idx], outputImage);
        }
        for (int p = 0; p < padding; p++) fputc(0, outputImage);
    }

    free(b); free(g); free(r);
    fclose(image);
    fclose(outputImage);
}

/**
 * Box blur en dos pasadas sobre imagen en escala de grises  (filtro "dg").
 *
 * Aplica el mismo algoritmo de desenfoque separable que desenfoque_color()
 * de bmp_filters_core.h, pero convierte la imagen a grises antes del blur.
 *
 * Diferencia con desenfoque_color():
 *   - desenfoque_color()  →  blur sobre imagen a color  (3 canales B, G, R).
 *   - desenfoque()        →  convierte a gris primero, luego blur  (1 canal replicado 3 veces).
 *
 * Algoritmo:
 *   1. Lee toda la imagen fila por fila en in_rows[][].
 *   2. In-place convierte cada píxel a gris sobreescribiendo los 3 bytes del canal
 *      con el mismo valor: in_rows[i][x*3] = in_rows[i][x*3+1] = in_rows[i][x*3+2] = gray.
 *   3. Pasada 1 — blur horizontal: promedia k vecinos en cada fila → tmp_rows[][].
 *   4. Pasada 2 — blur vertical:   promedia k vecinos en cada columna de tmp_rows[][] → out_rows[][].
 *   5. Escribe out_rows[][] al archivo de salida fila por fila.
 *
 * Fórmula de gris aplicada en el paso 2:
 *   gray = 0.21 * R + 0.72 * G + 0.07 * B
 *
 * Invocado desde bmp_processor.c dentro de:
 *   #pragma omp section  { if (filters.dg) desenfoque(image_paths[i], out_dg, kernel_gray); }
 *
 * input_path   Ruta del archivo BMP de entrada.
 * output_path  Ruta del archivo BMP de salida. Nombre: <original>_dg.bmp
 * kernel_size  Tamaño del kernel. Debe ser entero positivo e impar.
 *                     Recibido desde --kernel-gray en el CLI.
 *                     Radio efectivo: k = kernel_size / 2.
 */
static void desenfoque(const char *input_path, const char *output_path, int kernel_size) {
    FILE *image = fopen(input_path, "rb");
    if (!image) {
        return;
    }

    FILE *outputImage = fopen(output_path, "wb");
    if (!outputImage) {
        fclose(image);
        return;
    }

    int ancho, alto, offset;
    manejar_encabezado_1(image, outputImage, &ancho, &alto, &offset);

    /* row_padded: tamaño de fila en bytes, alineado a 4 bytes */
    int row_padded = (ancho * 3 + 3) & (~3);

    unsigned char **in_rows  = (unsigned char **)malloc(alto * sizeof(unsigned char *));
    unsigned char **out_rows = (unsigned char **)malloc(alto * sizeof(unsigned char *));
    unsigned char **tmp_rows = (unsigned char **)malloc(alto * sizeof(unsigned char *));

    /* Paso 1: Lee imagen y convierte a gris en in_rows */
    for (int i = 0; i < alto; i++) {
        in_rows[i]  = (unsigned char *)malloc(row_padded);
        out_rows[i] = (unsigned char *)malloc(row_padded);
        tmp_rows[i] = (unsigned char *)malloc(row_padded);
        fread(in_rows[i], 1, row_padded, image);

        /* Conversión in-place a gris: sobreescribe los 3 bytes del píxel con el valor de gris */
        for (int x = 0; x < ancho; x++) {
            unsigned char b    = in_rows[i][x * 3];
            unsigned char g    = in_rows[i][x * 3 + 1];
            unsigned char r    = in_rows[i][x * 3 + 2];
            unsigned char gray = (unsigned char)(0.21 * r + 0.72 * g + 0.07 * b);
            in_rows[i][x * 3]     = gray;
            in_rows[i][x * 3 + 1] = gray;
            in_rows[i][x * 3 + 2] = gray;
        }
    }

    int k = kernel_size / 2;  /* radio: k píxeles a cada lado del centro */

    /* Pasada 1 — Blur horizontal sobre in_rows → tmp_rows */
    for (int y = 0; y < alto; y++) {
        for (int x = 0; x < ancho; x++) {
            int sum = 0, count = 0;
            for (int dx = -k; dx <= k; dx++) {
                int nx = x + dx;
                if (nx >= 0 && nx < ancho) {
                    sum += in_rows[y][nx * 3];  /* canal gris = mismo valor en los 3 bytes */
                    count++;
                }
            }
            unsigned char blurred = (unsigned char)(sum / count);
            tmp_rows[y][x * 3]     = blurred;
            tmp_rows[y][x * 3 + 1] = blurred;
            tmp_rows[y][x * 3 + 2] = blurred;
        }
    }

    /* Pasada 2 — Blur vertical sobre tmp_rows → out_rows, escritura inmediata */
    for (int y = 0; y < alto; y++) {
        for (int x = 0; x < ancho; x++) {
            int sum = 0, count = 0;
            for (int dy = -k; dy <= k; dy++) {
                int ny = y + dy;
                if (ny >= 0 && ny < alto) {
                    sum += tmp_rows[ny][x * 3];
                    count++;
                }
            }
            unsigned char blurred = (unsigned char)(sum / count);
            out_rows[y][x * 3]     = blurred;
            out_rows[y][x * 3 + 1] = blurred;
            out_rows[y][x * 3 + 2] = blurred;
        }
        fwrite(out_rows[y], 1, row_padded, outputImage);
    }

    for (int i = 0; i < alto; i++) {
        free(in_rows[i]); free(tmp_rows[i]); free(out_rows[i]);
    }
    free(in_rows); free(tmp_rows); free(out_rows);
    fclose(image);
    fclose(outputImage);
}

#endif /* BMP_FILTERS_EXTENDED_H */