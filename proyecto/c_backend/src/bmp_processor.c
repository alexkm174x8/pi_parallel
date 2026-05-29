#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <sys/stat.h>

// compatibilidad multiplataforma para crear carpetas y separar rutas
#ifdef _WIN32
#include <direct.h>
#define MKDIR(path) _mkdir(path)
#define PATH_SEP '\\'
#else
#include <unistd.h>
#define MKDIR(path) mkdir(path, 0777)
#define PATH_SEP '/'
#endif

// funciones de filtros implementadas como header-only
#include "../include/bmp_filters_core.h"
#include "../include/bmp_filters_extended.h"

// configuracion global del ejecutable
#define NUM_THREADS 18
#define MAX_IMAGES 10
#define MAX_PATH_LEN 1024

// estructura booleana de filtros solicitados por cli
typedef struct {
    int vg;  // inversion vertical gris
    int vc;  // inversion vertical color
    int hg;  // espejo horizontal gris
    int hc;  // espejo horizontal color
    int dg;  // desenfoque gris
    int dc;  // desenfoque color
} FilterSelection;

// esta funcion muestra como usar el programa desde consola
static void print_usage(const char *program_name) {
    // mensaje de ayuda cuando faltan parametros o se usa --help
    fprintf(stderr,
            "Uso:\n"
            "  %s --output <carpeta> --filters <vg,vc,hg,hc,dg,dc> "
            "[--kernel-gray <impar>] [--kernel-color <impar>] <img1.bmp> [img2.bmp ...]\n",
            program_name);
}

// esta funcion valida rapido si un kernel cumple la regla basica
static int is_valid_kernel(int value) {
    // regla de negocio: kernel positivo e impar
    return value > 0 && value % 2 == 1;
}

// esta funcion revisa si una carpeta ya existe
static int directory_exists(const char *path) {
    // verifica si la ruta existe y corresponde a una carpeta
    struct stat info;
    return stat(path, &info) == 0 && (info.st_mode & S_IFDIR);
}

// esta funcion asegura que exista la carpeta de salida
static int ensure_directory(const char *path) {
    // intenta usar carpeta existente; si no existe, intenta crearla
    if (directory_exists(path)) {
        return 1;
    }

    if (MKDIR(path) == 0) {
        return 1;
    }

    // si alguien la creo en paralelo entre chequeo y creacion, tambien cuenta como exito
    return errno == EEXIST;
}

// esta funcion revisa si una ruta termina en bmp
static int ends_with_bmp(const char *path) {
    // validacion simple de extension, case-insensitive
    size_t length = strlen(path);
    if (length < 4) {
        return 0;
    }

    const char *ext = path + length - 4;
    return (tolower((unsigned char)ext[0]) == '.') &&
           (tolower((unsigned char)ext[1]) == 'b') &&
           (tolower((unsigned char)ext[2]) == 'm') &&
           (tolower((unsigned char)ext[3]) == 'p');
}

// esta funcion toma solo el nombre del archivo sin carpetas
static const char *file_basename(const char *path) {
    // obtiene solo nombre de archivo removiendo directorios previos
    // considera separadores unix (/) y windows (\)
    const char *slash = strrchr(path, '/');
    const char *backslash = strrchr(path, '\\');
    const char *base = path;

    if (slash && backslash) {
        base = (slash > backslash) ? slash + 1 : backslash + 1;
    } else if (slash) {
        base = slash + 1;
    } else if (backslash) {
        base = backslash + 1;
    }

    return base;
}

// esta funcion arma el nombre final de cada archivo de salida
static void build_output_path(char *buffer, size_t buffer_size, const char *output_dir,
                              const char *input_path, const char *suffix) {
    // construye ruta de salida:
    // <output_dir>/<nombre_original>_<sufijo>.bmp
    const char *base = file_basename(input_path);
    const char *dot = strrchr(base, '.');
    size_t name_len = dot ? (size_t)(dot - base) : strlen(base);
    snprintf(buffer, buffer_size, "%s%c%.*s_%s.bmp", output_dir, PATH_SEP, (int)name_len, base, suffix);
}

// esta funcion convierte la lista de filtros en banderas internas
static int parse_filters(const char *value, FilterSelection *filters) {
    // convierte cadena "vg,dc,hg" a flags dentro de filterselection
    // nota: strtok modifica buffer, por eso se copia la entrada
    char buffer[128];
    char *token = NULL;

    memset(filters, 0, sizeof(*filters));
    snprintf(buffer, sizeof(buffer), "%s", value);
    token = strtok(buffer, ",");

    while (token != NULL) {
        if (strcmp(token, "vg") == 0) {
            filters->vg = 1;
        } else if (strcmp(token, "vc") == 0) {
            filters->vc = 1;
        } else if (strcmp(token, "hg") == 0) {
            filters->hg = 1;
        } else if (strcmp(token, "hc") == 0) {
            filters->hc = 1;
        } else if (strcmp(token, "dg") == 0) {
            filters->dg = 1;
        } else if (strcmp(token, "dc") == 0) {
            filters->dc = 1;
        } else {
            fprintf(stderr, "Filtro no reconocido: %s\n", token);
            return 0;
        }

        token = strtok(NULL, ",");
    }

    // debe quedar al menos un filtro activo
    return filters->vg || filters->vc || filters->hg || filters->hc || filters->dg || filters->dc;
}

// esta funcion convierte texto a entero para leer kernels
static int parse_int(const char *value, int *target) {
    // parser robusto para enteros (1..999)
    char *end = NULL;
    long parsed = strtol(value, &end, 10);
    if (*value == '\0' || *end != '\0' || parsed <= 0 || parsed > 999) {
        return 0;
    }

    *target = (int)parsed;
    return 1;
}

// aqui empieza el flujo principal de lectura validacion y procesamiento
int main(int argc, char *argv[]) {
    // parametros de ejecucion
    const char *output_dir = NULL;
    const char *image_paths[MAX_IMAGES];
    int image_count = 0;
    int kernel_gray = 0;
    int kernel_color = 0;
    FilterSelection filters;

    memset(&filters, 0, sizeof(filters));

    // minimo practico de argumentos para una corrida valida
    if (argc < 5) {
        print_usage(argv[0]);
        return 1;
    }

    // parseo manual de argumentos estilo --flag valor
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--output") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Falta el valor para --output\n");
                return 1;
            }
            output_dir = argv[++i];
        } else if (strcmp(argv[i], "--filters") == 0) {
            if (i + 1 >= argc || !parse_filters(argv[++i], &filters)) {
                fprintf(stderr, "La lista de filtros es invalida\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--kernel-gray") == 0) {
            if (i + 1 >= argc || !parse_int(argv[++i], &kernel_gray)) {
                fprintf(stderr, "Kernel gris invalido\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--kernel-color") == 0) {
            if (i + 1 >= argc || !parse_int(argv[++i], &kernel_color)) {
                fprintf(stderr, "Kernel color invalido\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            // todo argumento no reconocido se interpreta como ruta de imagen
            if (image_count >= MAX_IMAGES) {
                fprintf(stderr, "Solo se permiten hasta %d imagenes por ejecucion\n", MAX_IMAGES);
                return 1;
            }
            image_paths[image_count++] = argv[i];
        }
    }

    // validaciones globales antes de procesar
    if (!output_dir || image_count == 0) {
        fprintf(stderr, "Debes indicar carpeta de salida y al menos una imagen BMP\n");
        return 1;
    }

    if (!ensure_directory(output_dir)) {
        fprintf(stderr, "No se pudo crear o acceder a la carpeta de salida: %s\n", output_dir);
        return 1;
    }

    for (int i = 0; i < image_count; i++) {
        if (!ends_with_bmp(image_paths[i])) {
            fprintf(stderr, "Archivo no valido (solo BMP): %s\n", image_paths[i]);
            return 1;
        }
    }

    if ((filters.dg && !is_valid_kernel(kernel_gray)) || (filters.dc && !is_valid_kernel(kernel_color))) {
        fprintf(stderr, "Los kernels de desenfoque deben ser enteros positivos e impares\n");
        return 1;
    }

    // inicia cronometro de ejecucion total
    omp_set_num_threads(NUM_THREADS);
    double inicio = omp_get_wtime();

    // procesa imagen por imagen
    for (int i = 0; i < image_count; i++) {
        // rutas de salida posibles para los 6 filtros
        char out_vg[MAX_PATH_LEN];
        char out_vc[MAX_PATH_LEN];
        char out_hg[MAX_PATH_LEN];
        char out_hc[MAX_PATH_LEN];
        char out_dg[MAX_PATH_LEN];
        char out_dc[MAX_PATH_LEN];

        build_output_path(out_vg, sizeof(out_vg), output_dir, image_paths[i], "vg");
        build_output_path(out_vc, sizeof(out_vc), output_dir, image_paths[i], "vc");
        build_output_path(out_hg, sizeof(out_hg), output_dir, image_paths[i], "hg");
        build_output_path(out_hc, sizeof(out_hc), output_dir, image_paths[i], "hc");
        build_output_path(out_dg, sizeof(out_dg), output_dir, image_paths[i], "dg");
        build_output_path(out_dc, sizeof(out_dc), output_dir, image_paths[i], "dc");

        // paralelismo por secciones:
        // cada seccion ejecuta un filtro independiente sobre la misma imagen
        // las secciones no comparten buffers entre si, por eso son seguras
#pragma omp parallel sections
        {
#pragma omp section
            {
                if (filters.vg) {
                    inv_img(out_vg, image_paths[i]);
                }
            }
#pragma omp section
            {
                if (filters.vc) {
                    inv_img_color(out_vc, image_paths[i]);
                }
            }
#pragma omp section
            {
                if (filters.hg) {
                    inv_img_grey_horizontal(out_hg, image_paths[i]);
                }
            }
#pragma omp section
            {
                if (filters.hc) {
                    inv_img_color_horizontal(out_hc, image_paths[i]);
                }
            }
#pragma omp section
            {
                if (filters.dg) {
                    desenfoque(image_paths[i], out_dg, kernel_gray);
                }
            }
#pragma omp section
            {
                if (filters.dc) {
                    desenfoque_color(image_paths[i], out_dc, kernel_color);
                }
            }
        }
    }

    // reporte de tiempo para que python lo parsee con regex
    double total = omp_get_wtime() - inicio;
    printf("TOTAL_TIME=%.6f\n", total);
    printf("OUTPUT_DIR=%s\n", output_dir);
    return 0;
}
