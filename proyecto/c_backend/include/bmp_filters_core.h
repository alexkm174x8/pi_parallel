/**
 * bmp_filters_core.h
 * - Filtros principales de procesamiento de imágenes BMP.
 *
 * Implementa tres transformaciones sobre archivos BMP de 24 bits (3 bytes por píxel):
 *
 *   - inv_img()          Inversión vertical + conversión a escala de grises  (filtro "vg")
 *   - inv_img_color()    Inversión vertical conservando colores originales    (filtro "vc")
 *   - desenfoque_color() Box blur en dos pasadas sobre imagen a color         (filtro "dc")
 *
 * Todas las funciones son `static` (no exportadas) y se incluyen directamente
 * en bmp_processor.c mediante #include.  Se invocan dentro de un bloque
 * `#pragma omp parallel sections`, por lo que cada función corre en su propio hilo
 * de OpenMP de forma concurrente con las funciones de bmp_filters_extended.h.
 *
 * Formato BMP asumido:
 *   - 24 bits por píxel (3 bytes: B, G, R en ese orden según la especificación BMP).
 *   - Padding por fila: cada fila de píxeles se rellena a múltiplo de 4 bytes.
 *   - El encabezado completo (14 bytes de archivo + DIB header) se copia íntegro
 *     al archivo de salida sin modificación.
 *
 * Dependencias: <stdio.h>, <stdlib.h>, <string.h>
 */

#ifndef BMP_FILTERS_CORE_H
#define BMP_FILTERS_CORE_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


/* ===========================================================================
 * Utilidades internas
 * =========================================================================*/

/**
 * Calcula el padding (bytes de relleno) al final de cada fila BMP.
 *
 * El formato BMP requiere que cada fila de píxeles ocupe un número de bytes
 * múltiplo de 4.  Para una imagen de 24 bpp, cada píxel ocupa 3 bytes, por
 * lo que la fila tiene (ancho * 3) bytes de datos más 0-3 bytes de padding.
 *
 * Fórmula: padding = (4 - (ancho * 3) % 4) % 4
 *   - Si (ancho * 3) ya es múltiplo de 4, el resultado es 0.
 *
 * param ancho -  Ancho de la imagen en píxeles.
 * retorna -        Número de bytes de padding por fila (0, 1, 2 o 3).
 */
static inline int calcular_padding(int ancho) {
    return (4 - (ancho * 3) % 4) % 4;
}

/**
 * Lee y copia el encabezado BMP completo de entrada a salida.
 *
 * Lee los 14 bytes del file header y el DIB header (cuyo tamaño varía;
 * se determina a partir del campo offset en el file header).
 * Escribe el encabezado completo en el archivo de salida sin modificación,
 * y extrae el ancho, alto y offset de datos de píxeles para uso posterior.
 *
 * Campos extraídos del encabezado BMP:
 *   - offset: byte 10-13 del file header (inicio de datos de píxeles).
 *   - ancho:  bytes 18-21 del DIB header (BITMAPINFOHEADER.biWidth).
 *   - alto:   bytes 22-25 del DIB header (BITMAPINFOHEADER.biHeight).
 *
 * in      Archivo BMP de entrada abierto en modo "rb".
 * out     Archivo BMP de salida abierto en modo "wb".
 * ancho   [out] Ancho de la imagen en píxeles.
 * alto    [out] Alto de la imagen en píxeles.
 * offset  [out] Posición en bytes donde comienzan los datos de píxeles.
 */
static void manejar_encabezado(FILE *in, FILE *out, int *ancho, int *alto, int *offset) {
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


/* ===========================================================================
 * Filtros de imagen
 * =========================================================================*/

/**
 * Inversión vertical con conversión a escala de grises  (filtro "vg").
 *
 * Voltea la imagen verticalmente (la fila superior pasa a ser la inferior y
 * viceversa) y convierte cada píxel a un valor de gris usando la fórmula
 * de luminancia perceptual:
 *
 *   gray = 0.21 * R + 0.72 * G + 0.07 * B
 *
 * Los coeficientes reflejan la sensibilidad del ojo humano a cada canal.
 * El píxel gris se escribe tres veces (B=G=R=gray) para mantener el formato
 * BMP de 24 bpp.
 *
 * Algoritmo:
 *   1. Lee todas las filas de la imagen original en un arreglo de punteros
 *      (una entrada por fila), convirtiendo cada píxel a gris en el momento.
 *   2. Escribe las filas en orden inverso (de alto-1 a 0) al archivo de salida.
 *
 * Invocado desde bmp_processor.c dentro de:
 *   #pragma omp section  { if (filters.vg) inv_img(out_vg, image_paths[i]); }
 *
 * output_path  Ruta del archivo BMP de salida (se crea o sobreescribe).
 *                     Nombre esperado: <original>_vg.bmp
 * input_path   Ruta del archivo BMP de entrada.
 */
static void inv_img(const char *output_path, const char *input_path) {
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
    manejar_encabezado(image, outputImage, &ancho, &alto, &offset);

    int padding = calcular_padding(ancho);

    /* Reserva un arreglo de punteros, uno por fila */
    unsigned char **lineas = (unsigned char **)malloc(alto * sizeof(unsigned char *));

    /* Lectura: convierte cada píxel a gris durante la lectura */
    for (int i = 0; i < alto; i++) {
        lineas[i] = (unsigned char *)malloc(ancho);
        for (int j = 0; j < ancho; j++) {
            unsigned char b = fgetc(image);
            unsigned char g = fgetc(image);
            unsigned char r = fgetc(image);
            /* Fórmula de luminancia perceptual */
            lineas[i][j] = (unsigned char)(0.21 * r + 0.72 * g + 0.07 * b);
        }
        /* Descarta los bytes de padding de la fila actual */
        for (int p = 0; p < padding; p++) fgetc(image);
    }

    /* Escritura en orden inverso: última fila primero */
    for (int i = alto - 1; i >= 0; i--) {
        for (int x = 0; x < ancho; x++) {
            unsigned char pixel = lineas[i][x];
            /* BMP 24bpp: escribe B, G, R; los tres iguales produce gris */
            fputc(pixel, outputImage);
            fputc(pixel, outputImage);
            fputc(pixel, outputImage);
        }
        for (int p = 0; p < padding; p++) fputc(0, outputImage);
    }

    for (int i = 0; i < alto; i++) free(lineas[i]);
    free(lineas);
    fclose(image);
    fclose(outputImage);
}

/**
 * Inversión vertical conservando el color original  (filtro "vc").
 *
 * Voltea la imagen verticalmente igual que inv_img(), pero mantiene los
 * canales R, G, B intactos.  La imagen de salida tiene los mismos colores
 * que la original pero con las filas en orden invertido.
 *
 * Algoritmo:
 *   1. Lee todas las filas separando cada canal (B, G, R) en tres arreglos
 *      independientes para evitar entrelazado durante la escritura.
 *   2. Escribe las filas en orden inverso preservando los valores RGB.
 *
 * Invocado desde bmp_processor.c dentro de:
 *   #pragma omp section  { if (filters.vc) inv_img_color(out_vc, image_paths[i]); }
 *
 * output_path  Ruta del archivo BMP de salida. Nombre: <original>_vc.bmp
 * input_path   Ruta del archivo BMP de entrada.
 */
static void inv_img_color(const char *output_path, const char *input_path) {
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
    manejar_encabezado(image, outputImage, &ancho, &alto, &offset);

    int padding = calcular_padding(ancho);

    /* Tres arreglos separados, uno por canal de color */
    unsigned char **b_lineas = (unsigned char **)malloc(alto * sizeof(unsigned char *));
    unsigned char **g_lineas = (unsigned char **)malloc(alto * sizeof(unsigned char *));
    unsigned char **r_lineas = (unsigned char **)malloc(alto * sizeof(unsigned char *));

    /* Lectura: almacena cada canal por separado */
    for (int i = 0; i < alto; i++) {
        b_lineas[i] = (unsigned char *)malloc(ancho);
        g_lineas[i] = (unsigned char *)malloc(ancho);
        r_lineas[i] = (unsigned char *)malloc(ancho);
        for (int j = 0; j < ancho; j++) {
            b_lineas[i][j] = fgetc(image);  /* BMP almacena en orden B, G, R */
            g_lineas[i][j] = fgetc(image);
            r_lineas[i][j] = fgetc(image);
        }
        for (int p = 0; p < padding; p++) fgetc(image);
    }

    /* Escritura en orden inverso con colores intactos */
    for (int i = alto - 1; i >= 0; i--) {
        for (int y = 0; y < ancho; y++) {
            fputc(b_lineas[i][y], outputImage);
            fputc(g_lineas[i][y], outputImage);
            fputc(r_lineas[i][y], outputImage);
        }
        for (int p = 0; p < padding; p++) fputc(0, outputImage);
    }

    for (int i = 0; i < alto; i++) {
        free(b_lineas[i]);
        free(g_lineas[i]);
        free(r_lineas[i]);
    }
    free(b_lineas); free(g_lineas); free(r_lineas);
    fclose(image);
    fclose(outputImage);
}

/**
 * Box blur en dos pasadas sobre imagen a color  (filtro "dc").
 *
 * Aplica un desenfoque de caja (box blur) separable en dos pasadas sucesivas:
 *   Pasada 1 — horizontal: promedia kernel_size píxeles consecutivos en cada fila.
 *   Pasada 2 — vertical:   promedia kernel_size píxeles consecutivos en cada columna.
 *
 * El resultado se almacena en una imagen intermedia (temp_rows) para que la
 * pasada vertical use los valores ya suavizados horizontalmente.
 *
 * El tamaño efectivo del vecindario es kernel_size × kernel_size píxeles.
 * Para un kernel de tamaño k, cada píxel se promedia con los k/2 vecinos
 * de cada lado (con manejo de bordes: se ignoran los índices fuera del rango).
 *
 * La imagen de salida conserva el color (no convierte a gris).
 *
 * Fórmula por canal en la pasada horizontal:
 *   temp[y][x] = mean(input[y][x-k..x+k])  para cada canal B, G, R
 *
 * Invocado desde bmp_processor.c dentro de:
 *   #pragma omp section  { if (filters.dc) desenfoque_color(image_paths[i], out_dc, kernel_color); }
 *
 * input_path   Ruta del archivo BMP de entrada.
 * output_path  Ruta del archivo BMP de salida. Nombre: <original>_dc.bmp
 * kernel_size  Tamaño del kernel de desenfoque. Debe ser entero positivo
 *                     e impar (validado en la GUI antes de llegar aquí).
 *                     Valor recibido desde --kernel-color en el CLI.
 */
static void desenfoque_color(const char *input_path, const char *output_path, int kernel_size) {
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
    manejar_encabezado(image, outputImage, &ancho, &alto, &offset);

    /* row_padded: bytes por fila incluyendo padding, alineado a 4 bytes */
    int row_padded = (ancho * 3 + 3) & (~3);

    unsigned char **input_rows  = (unsigned char **)malloc(alto * sizeof(unsigned char *));
    unsigned char **output_rows = (unsigned char **)malloc(alto * sizeof(unsigned char *));
    unsigned char **temp_rows   = (unsigned char **)malloc(alto * sizeof(unsigned char *));

    /* Carga toda la imagen en memoria */
    for (int i = 0; i < alto; i++) {
        input_rows[i]  = (unsigned char *)malloc(row_padded);
        output_rows[i] = (unsigned char *)malloc(row_padded);
        temp_rows[i]   = (unsigned char *)malloc(row_padded);
        fread(input_rows[i], 1, row_padded, image);
    }

    int k = kernel_size / 2;  /* radio del kernel: píxeles a cada lado del centro */

    /* Pasada 1 — Blur horizontal (por canal B, G, R) */
    for (int y = 0; y < alto; y++) {
        for (int x = 0; x < ancho; x++) {
            int sB = 0, sG = 0, sR = 0, count = 0;
            for (int dx = -k; dx <= k; dx++) {
                int nx = x + dx;
                if (nx >= 0 && nx < ancho) {   /* manejo de borde: omite fuera de rango */
                    sB += input_rows[y][nx * 3];
                    sG += input_rows[y][nx * 3 + 1];
                    sR += input_rows[y][nx * 3 + 2];
                    count++;
                }
            }
            temp_rows[y][x * 3]     = (unsigned char)(sB / count);
            temp_rows[y][x * 3 + 1] = (unsigned char)(sG / count);
            temp_rows[y][x * 3 + 2] = (unsigned char)(sR / count);
        }
    }

    /* Pasada 2 — Blur vertical sobre el resultado de temp_rows */
    for (int y = 0; y < alto; y++) {
        for (int x = 0; x < ancho; x++) {
            int sB = 0, sG = 0, sR = 0, count = 0;
            for (int dy = -k; dy <= k; dy++) {
                int ny = y + dy;
                if (ny >= 0 && ny < alto) {
                    sB += temp_rows[ny][x * 3];
                    sG += temp_rows[ny][x * 3 + 1];
                    sR += temp_rows[ny][x * 3 + 2];
                    count++;
                }
            }
            output_rows[y][x * 3]     = (unsigned char)(sB / count);
            output_rows[y][x * 3 + 1] = (unsigned char)(sG / count);
            output_rows[y][x * 3 + 2] = (unsigned char)(sR / count);
        }
        fwrite(output_rows[y], 1, row_padded, outputImage);  /* escribe fila a disco */
    }

    for (int i = 0; i < alto; i++) {
        free(input_rows[i]); free(temp_rows[i]); free(output_rows[i]);
    }
    free(input_rows); free(temp_rows); free(output_rows);
    fclose(image);
    fclose(outputImage);
}

#endif