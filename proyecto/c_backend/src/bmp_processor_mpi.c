/*
 * MPI-enabled backend for BMP Parallel Studio
 *
 * This program implements a simple master-worker dynamic scheduler using MPI.
 * The master (rank 0) parses command-line arguments, builds a list of
 * (image,filter,kernel) tasks and distributes them to worker ranks.
 * Workers receive tasks, execute the corresponding filter (using the
 * existing filter implementations), verify the output file was created
 * and report back success/failure.
 *
 * The design assumes a shared filesystem: input image paths and the
 * output directory must be visible at the same path from all nodes.
 */

#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <direct.h>
#define MKDIR(path) _mkdir(path)
#define PATH_SEP '\\'
#else
#include <unistd.h>
#define MKDIR(path) mkdir(path, 0777)
#define PATH_SEP '/'
#endif

#include "../include/bmp_filters_core.h"
#include "../include/bmp_filters_extended.h"

#define MAX_IMAGES 10
#define MAX_PATH_LEN 1024
#define BUF_SIZE 4096
#define NODE_NAME_LEN 256

typedef struct {
    int vg;
    int vc;
    int hg;
    int hc;
    int dg;
    int dc;
} FilterSelection;

static void print_usage(const char *program_name) {
    fprintf(stderr,
            "Uso (MPI):\n"
            "  mpirun -np <N> %s --output <carpeta> --filters <vg,vc,hg,hc,dg,dc> "
            "[--kernel-gray <impar>] [--kernel-color <impar>] <img1.bmp> [img2.bmp ...]\n",
            program_name);
}

static int is_valid_kernel(int value) {
    return value > 0 && value % 2 == 1;
}

static int directory_exists(const char *path) {
    struct stat info;
    return stat(path, &info) == 0 && (info.st_mode & S_IFDIR);
}

static int ensure_directory(const char *path) {
    if (directory_exists(path)) {
        return 1;
    }

    if (MKDIR(path) == 0) {
        return 1;
    }

    return errno == EEXIST;
}

static int ends_with_bmp(const char *path) {
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

static const char *file_basename(const char *path) {
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

static void build_output_path(char *buffer, size_t buffer_size, const char *output_dir,
                              const char *input_path, const char *suffix) {
    const char *base = file_basename(input_path);
    const char *dot = strrchr(base, '.');
    size_t name_len = dot ? (size_t)(dot - base) : strlen(base);
    snprintf(buffer, buffer_size, "%s%c%.*s_%s.bmp", output_dir, PATH_SEP, (int)name_len, base, suffix);
}

static int parse_filters(const char *value, FilterSelection *filters) {
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

    return filters->vg || filters->vc || filters->hg || filters->hc || filters->dg || filters->dc;
}

static int parse_int(const char *value, int *target) {
    char *end = NULL;
    long parsed = strtol(value, &end, 10);
    if (*value == '\0' || *end != '\0' || parsed <= 0 || parsed > 999) {
        return 0;
    }

    *target = (int)parsed;
    return 1;
}

static void execute_task(const char *task, const char *output_dir, char *result_msg, size_t result_msg_size) {
    char task_copy[BUF_SIZE];
    char output_path[MAX_PATH_LEN];

    snprintf(task_copy, sizeof(task_copy), "%s", task);

    char *saveptr = NULL;
    char *input_path = strtok_r(task_copy, "\t", &saveptr);
    char *filter = strtok_r(NULL, "\t", &saveptr);
    char *kernel_s = strtok_r(NULL, "\t", &saveptr);

    if (!input_path || !filter || !kernel_s) {
        snprintf(result_msg, result_msg_size, "ERR\tinvalid_task\t%s", task);
        return;
    }

    int kernel = atoi(kernel_s);
    build_output_path(output_path, sizeof(output_path), output_dir, input_path, filter);

    if (!directory_exists(output_dir)) {
        snprintf(result_msg, result_msg_size, "ERR\toutput_dir_inaccessible\t%s", output_dir);
        return;
    }

    struct stat st_in;
    if (stat(input_path, &st_in) != 0) {
        snprintf(result_msg, result_msg_size, "ERR\tinput_inaccessible\t%s", input_path);
        return;
    }

    if (strcmp(filter, "vg") == 0) {
        inv_img(output_path, input_path);
    } else if (strcmp(filter, "vc") == 0) {
        inv_img_color(output_path, input_path);
    } else if (strcmp(filter, "hg") == 0) {
        inv_img_grey_horizontal(output_path, input_path);
    } else if (strcmp(filter, "hc") == 0) {
        inv_img_color_horizontal(output_path, input_path);
    } else if (strcmp(filter, "dg") == 0) {
        desenfoque(input_path, output_path, kernel);
    } else if (strcmp(filter, "dc") == 0) {
        desenfoque_color(input_path, output_path, kernel);
    } else {
        snprintf(result_msg, result_msg_size, "ERR\tunknown_filter\t%s", filter);
        return;
    }

    struct stat st_out;
    if (stat(output_path, &st_out) != 0) {
        snprintf(result_msg, result_msg_size, "ERR\tno_output\t%s", output_path);
        return;
    }

    snprintf(result_msg, result_msg_size, "OK\t%s", output_path);
}

static int is_error_result(const char *result_msg) {
    const char *err_pos = strstr(result_msg, "ERR\t");
    return err_pos != NULL;
}

static void parse_task_for_log(const char *task, char *input_path, size_t input_path_size, char *filter, size_t filter_size) {
    char task_copy[BUF_SIZE];
    char *saveptr = NULL;
    char *img = NULL;
    char *flt = NULL;

    snprintf(task_copy, sizeof(task_copy), "%s", task);
    img = strtok_r(task_copy, "\t", &saveptr);
    flt = strtok_r(NULL, "\t", &saveptr);

    if (img) {
        snprintf(input_path, input_path_size, "%s", img);
    } else {
        input_path[0] = '\0';
    }

    if (flt) {
        snprintf(filter, filter_size, "%s", flt);
    } else {
        filter[0] = '\0';
    }
}

static void print_dispatch_log(int src_rank, const char *src_name, int dst_rank, const char *dst_name,
                               const char *task, const char *output_dir) {
    char input_path[MAX_PATH_LEN];
    char filter[32];
    char output_path[MAX_PATH_LEN];

    parse_task_for_log(task, input_path, sizeof(input_path), filter, sizeof(filter));
    build_output_path(output_path, sizeof(output_path), output_dir, input_path, filter);
    printf("DISPATCH source_rank=%d source_node=%s target_rank=%d target_node=%s image=%s filter=%s output=%s\n",
           src_rank, src_name, dst_rank, dst_name, input_path, filter, output_path);
    fflush(stdout);
}

static void print_complete_log(int rank, const char *node_name, const char *result_msg) {
    printf("COMPLETE rank=%d node=%s %s\n", rank, node_name, result_msg);
    fflush(stdout);
}

static void wrap_result_with_meta(const char *node_name, const char *task, const char *raw_result,
                                  double seconds, char *wrapped_result, size_t wrapped_size) {
    char input_path[MAX_PATH_LEN];
    char filter[32];
    parse_task_for_log(task, input_path, sizeof(input_path), filter, sizeof(filter));
    snprintf(wrapped_result, wrapped_size,
             "worker=%s\timage=%s\tfilter=%s\tseconds=%.6f\t%s",
             node_name, input_path, filter, seconds, raw_result);
}

int main(int argc, char *argv[]) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char node_name[NODE_NAME_LEN];
    int node_name_len = 0;
    MPI_Get_processor_name(node_name, &node_name_len);
    if (node_name_len < 0 || node_name_len >= NODE_NAME_LEN) {
        node_name[NODE_NAME_LEN - 1] = '\0';
    } else {
        node_name[node_name_len] = '\0';
    }

    char *all_node_names = NULL;
    if (rank == 0) {
        all_node_names = (char *)malloc((size_t)size * NODE_NAME_LEN);
    }
    MPI_Gather(node_name, NODE_NAME_LEN, MPI_CHAR, all_node_names, NODE_NAME_LEN, MPI_CHAR, 0, MPI_COMM_WORLD);

    const char *output_dir = NULL;
    const char *image_paths[MAX_IMAGES];
    int image_count = 0;
    int kernel_gray = 0;
    int kernel_color = 0;
    FilterSelection filters;

    memset(&filters, 0, sizeof(filters));

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0) {
            if (rank == 0) {
                print_usage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        }
    }

    /* Only rank 0 parses and validates arguments */
    if (rank == 0) {
        if (argc < 5) {
            print_usage(argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }

        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--output") == 0) {
                if (i + 1 >= argc) {
                    fprintf(stderr, "Falta el valor para --output\n");
                    MPI_Abort(MPI_COMM_WORLD, 1);
                    return 1;
                }
                output_dir = argv[++i];
            } else if (strcmp(argv[i], "--filters") == 0) {
                if (i + 1 >= argc || !parse_filters(argv[++i], &filters)) {
                    fprintf(stderr, "La lista de filtros es invalida\n");
                    MPI_Abort(MPI_COMM_WORLD, 1);
                    return 1;
                }
            } else if (strcmp(argv[i], "--kernel-gray") == 0) {
                if (i + 1 >= argc || !parse_int(argv[++i], &kernel_gray)) {
                    fprintf(stderr, "Kernel gris invalido\n");
                    MPI_Abort(MPI_COMM_WORLD, 1);
                    return 1;
                }
            } else if (strcmp(argv[i], "--kernel-color") == 0) {
                if (i + 1 >= argc || !parse_int(argv[++i], &kernel_color)) {
                    fprintf(stderr, "Kernel color invalido\n");
                    MPI_Abort(MPI_COMM_WORLD, 1);
                    return 1;
                }
            } else {
                if (image_count >= MAX_IMAGES) {
                    fprintf(stderr, "Solo se permiten hasta %d imagenes por ejecucion\n", MAX_IMAGES);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                    return 1;
                }
                image_paths[image_count++] = argv[i];
            }
        }

        if (!output_dir || image_count == 0) {
            fprintf(stderr, "Debes indicar carpeta de salida y al menos una imagen BMP\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }

        if (!ensure_directory(output_dir)) {
            fprintf(stderr, "No se pudo crear o acceder a la carpeta de salida: %s\n", output_dir);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }

        for (int i = 0; i < image_count; i++) {
            if (!ends_with_bmp(image_paths[i])) {
                fprintf(stderr, "Archivo no valido (solo BMP): %s\n", image_paths[i]);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
            /* ensure the input file exists on the shared filesystem */
            struct stat st;
            if (stat(image_paths[i], &st) != 0) {
                fprintf(stderr, "Archivo no encontrado o inaccesible: %s\n", image_paths[i]);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }

        if ((filters.dg && !is_valid_kernel(kernel_gray)) || (filters.dc && !is_valid_kernel(kernel_color))) {
            fprintf(stderr, "Los kernels de desenfoque deben ser enteros positivos e impares\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }

    /* Broadcast output_dir to all ranks (fixed-size buffer) */
    char output_dir_buf[MAX_PATH_LEN];
    if (rank == 0) {
        strncpy(output_dir_buf, output_dir, MAX_PATH_LEN - 1);
        output_dir_buf[MAX_PATH_LEN - 1] = '\0';
    } else {
        output_dir_buf[0] = '\0';
    }
    MPI_Bcast(output_dir_buf, MAX_PATH_LEN, MPI_CHAR, 0, MPI_COMM_WORLD);

    /* Broadcast kernels and filters as a small struct */
    int kernel_info[2];
    if (rank == 0) {
        kernel_info[0] = kernel_gray;
        kernel_info[1] = kernel_color;
    }
    MPI_Bcast(kernel_info, 2, MPI_INT, 0, MPI_COMM_WORLD);
    kernel_gray = kernel_info[0];
    kernel_color = kernel_info[1];

    int filters_buf[6] = {0,0,0,0,0,0};
    if (rank == 0) {
        filters_buf[0] = filters.vg;
        filters_buf[1] = filters.vc;
        filters_buf[2] = filters.hg;
        filters_buf[3] = filters.hc;
        filters_buf[4] = filters.dg;
        filters_buf[5] = filters.dc;
    }
    MPI_Bcast(filters_buf, 6, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        filters.vg = filters_buf[0];
        filters.vc = filters_buf[1];
        filters.hg = filters_buf[2];
        filters.hc = filters_buf[3];
        filters.dg = filters_buf[4];
        filters.dc = filters_buf[5];
    }

    /* Broadcast image count and image paths to workers */
    if (rank == 0) {
        MPI_Bcast(&image_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
        for (int i = 0; i < image_count; i++) {
            char tmp[MAX_PATH_LEN];
            strncpy(tmp, image_paths[i], MAX_PATH_LEN - 1);
            tmp[MAX_PATH_LEN - 1] = '\0';
            MPI_Bcast(tmp, MAX_PATH_LEN, MPI_CHAR, 0, MPI_COMM_WORLD);
        }
    } else {
        MPI_Bcast(&image_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
        for (int i = 0; i < image_count; i++) {
            char tmp[MAX_PATH_LEN];
            MPI_Bcast(tmp, MAX_PATH_LEN, MPI_CHAR, 0, MPI_COMM_WORLD);
            /* store into image_paths array allocated only on rank 0 previously; workers don't need to store */
            /* We won't use this array on workers since tasks are sent by master. */
        }
    }

    /* If only one MPI process is present, fall back to local sequential processing. */
    if (size == 1) {
        /* replicate local behaviour (simple): process images sequentially using selected filters */
        double start = MPI_Wtime();
        for (int i = 0; i < image_count; i++) {
            const char *img = image_paths[i];
            char out_vg[MAX_PATH_LEN];
            char out_vc[MAX_PATH_LEN];
            char out_hg[MAX_PATH_LEN];
            char out_hc[MAX_PATH_LEN];
            char out_dg[MAX_PATH_LEN];
            char out_dc[MAX_PATH_LEN];

            build_output_path(out_vg, sizeof(out_vg), output_dir_buf, img, "vg");
            build_output_path(out_vc, sizeof(out_vc), output_dir_buf, img, "vc");
            build_output_path(out_hg, sizeof(out_hg), output_dir_buf, img, "hg");
            build_output_path(out_hc, sizeof(out_hc), output_dir_buf, img, "hc");
            build_output_path(out_dg, sizeof(out_dg), output_dir_buf, img, "dg");
            build_output_path(out_dc, sizeof(out_dc), output_dir_buf, img, "dc");

            if (filters.vg) inv_img(out_vg, img);
            if (filters.vc) inv_img_color(out_vc, img);
            if (filters.hg) inv_img_grey_horizontal(out_hg, img);
            if (filters.hc) inv_img_color_horizontal(out_hc, img);
            if (filters.dg) desenfoque(img, out_dg, kernel_gray);
            if (filters.dc) desenfoque_color(img, out_dc, kernel_color);
        }
        double total = MPI_Wtime() - start;
        printf("TOTAL_TIME=%.6f\n", total);
        printf("OUTPUT_DIR=%s\n", output_dir_buf);
        if (all_node_names) {
            free(all_node_names);
        }
        MPI_Finalize();
        return 0;
    }

    /* Build task list on master (rank 0) */
    char **tasks = NULL;
    int total_tasks = 0;
    if (rank == 0) {
        /* count tasks */
        for (int i = 0; i < image_count; i++) {
            if (filters.vg) total_tasks++;
            if (filters.vc) total_tasks++;
            if (filters.hg) total_tasks++;
            if (filters.hc) total_tasks++;
            if (filters.dg) total_tasks++;
            if (filters.dc) total_tasks++;
        }

        if (total_tasks == 0) {
            fprintf(stderr, "No hay tareas para ejecutar.\n");
            MPI_Finalize();
            return 1;
        }

        tasks = (char **)malloc(sizeof(char *) * total_tasks);
        int idx = 0;
        for (int i = 0; i < image_count; i++) {
            const char *img = image_paths[i];
            if (filters.vg) {
                tasks[idx] = (char *)malloc(BUF_SIZE);
                snprintf(tasks[idx], BUF_SIZE, "%s\t%s\t%d", img, "vg", 0);
                idx++;
            }
            if (filters.vc) {
                tasks[idx] = (char *)malloc(BUF_SIZE);
                snprintf(tasks[idx], BUF_SIZE, "%s\t%s\t%d", img, "vc", 0);
                idx++;
            }
            if (filters.hg) {
                tasks[idx] = (char *)malloc(BUF_SIZE);
                snprintf(tasks[idx], BUF_SIZE, "%s\t%s\t%d", img, "hg", 0);
                idx++;
            }
            if (filters.hc) {
                tasks[idx] = (char *)malloc(BUF_SIZE);
                snprintf(tasks[idx], BUF_SIZE, "%s\t%s\t%d", img, "hc", 0);
                idx++;
            }
            if (filters.dg) {
                tasks[idx] = (char *)malloc(BUF_SIZE);
                snprintf(tasks[idx], BUF_SIZE, "%s\t%s\t%d", img, "dg", kernel_gray);
                idx++;
            }
            if (filters.dc) {
                tasks[idx] = (char *)malloc(BUF_SIZE);
                snprintf(tasks[idx], BUF_SIZE, "%s\t%s\t%d", img, "dc", kernel_color);
                idx++;
            }
        }
    }

    /* Timing starts here on master */
    double start_time = 0.0;
    if (rank == 0) start_time = MPI_Wtime();

    const int TAG_TASK = 1;
    const int TAG_RESULT = 2;

    if (rank == 0) {
        int tasks_sent = 0;
        int tasks_completed = 0;
        int failed_tasks = 0;

        /* Initial distribution: send one task to each worker (if available) */
        for (int dest = 1; dest < size; dest++) {
            if (tasks_sent < total_tasks) {
                print_dispatch_log(0, node_name, dest, all_node_names + (dest * NODE_NAME_LEN), tasks[tasks_sent], output_dir_buf);
                MPI_Send(tasks[tasks_sent], (int)strlen(tasks[tasks_sent]) + 1, MPI_CHAR, dest, TAG_TASK, MPI_COMM_WORLD);
                tasks_sent++;
            } else {
                MPI_Send("STOP", 5, MPI_CHAR, dest, TAG_TASK, MPI_COMM_WORLD);
            }
        }

        /* Receive results and send new tasks until all are done */
        while (tasks_completed < total_tasks) {
            char result_buf[BUF_SIZE];
            MPI_Status status;

            int has_result = 0;
            MPI_Iprobe(MPI_ANY_SOURCE, TAG_RESULT, MPI_COMM_WORLD, &has_result, &status);

            if (has_result) {
                MPI_Recv(result_buf, BUF_SIZE, MPI_CHAR, status.MPI_SOURCE, TAG_RESULT, MPI_COMM_WORLD, &status);
                tasks_completed++;

                if (is_error_result(result_buf)) {
                    failed_tasks++;
                    fprintf(stderr, "MPI task failed on rank %d: %s\n", status.MPI_SOURCE, result_buf);
                }
                print_complete_log(status.MPI_SOURCE, all_node_names + (status.MPI_SOURCE * NODE_NAME_LEN), result_buf);

                if (tasks_sent < total_tasks) {
                    print_dispatch_log(0, node_name, status.MPI_SOURCE,
                                       all_node_names + (status.MPI_SOURCE * NODE_NAME_LEN), tasks[tasks_sent], output_dir_buf);
                    MPI_Send(tasks[tasks_sent], (int)strlen(tasks[tasks_sent]) + 1, MPI_CHAR, status.MPI_SOURCE, TAG_TASK, MPI_COMM_WORLD);
                    tasks_sent++;
                } else {
                    MPI_Send("STOP", 5, MPI_CHAR, status.MPI_SOURCE, TAG_TASK, MPI_COMM_WORLD);
                }

                continue;
            }

            if (tasks_sent < total_tasks) {
                print_dispatch_log(0, node_name, 0, node_name, tasks[tasks_sent], output_dir_buf);
                double task_start = MPI_Wtime();
                execute_task(tasks[tasks_sent], output_dir_buf, result_buf, sizeof(result_buf));
                double task_seconds = MPI_Wtime() - task_start;
                char wrapped_result[BUF_SIZE];
                wrap_result_with_meta(node_name, tasks[tasks_sent], result_buf, task_seconds, wrapped_result, sizeof(wrapped_result));
                snprintf(result_buf, sizeof(result_buf), "%s", wrapped_result);
                tasks_sent++;
                tasks_completed++;

                if (is_error_result(result_buf)) {
                    failed_tasks++;
                    fprintf(stderr, "MPI task failed on rank 0: %s\n", result_buf);
                }
                print_complete_log(0, node_name, result_buf);
            } else {
                MPI_Recv(result_buf, BUF_SIZE, MPI_CHAR, MPI_ANY_SOURCE, TAG_RESULT, MPI_COMM_WORLD, &status);
                tasks_completed++;

                if (is_error_result(result_buf)) {
                    failed_tasks++;
                    fprintf(stderr, "MPI task failed on rank %d: %s\n", status.MPI_SOURCE, result_buf);
                }
                print_complete_log(status.MPI_SOURCE, all_node_names + (status.MPI_SOURCE * NODE_NAME_LEN), result_buf);

                MPI_Send("STOP", 5, MPI_CHAR, status.MPI_SOURCE, TAG_TASK, MPI_COMM_WORLD);
            }
        }

        double total = MPI_Wtime() - start_time;
        printf("TOTAL_TIME=%.6f\n", total);
        printf("OUTPUT_DIR=%s\n", output_dir_buf);

        /* free tasks */
        for (int i = 0; i < total_tasks; i++) {
            free(tasks[i]);
        }
        free(tasks);
        if (all_node_names) {
            free(all_node_names);
        }

        if (failed_tasks > 0) {
            fprintf(stderr, "MPI finished with %d failed task(s).\n", failed_tasks);
            MPI_Finalize();
            return 1;
        }

    } else {
        /* Worker loop */
        while (1) {
            char task_buf[BUF_SIZE];
            MPI_Status status;
            MPI_Recv(task_buf, BUF_SIZE, MPI_CHAR, 0, TAG_TASK, MPI_COMM_WORLD, &status);
            if (strcmp(task_buf, "STOP") == 0) {
                break;
            }

            char result_msg[BUF_SIZE];
            double task_start = MPI_Wtime();
            execute_task(task_buf, output_dir_buf, result_msg, sizeof(result_msg));
            double task_seconds = MPI_Wtime() - task_start;

            char wrapped_result[BUF_SIZE];
            wrap_result_with_meta(node_name, task_buf, result_msg, task_seconds, wrapped_result, sizeof(wrapped_result));
            MPI_Send(wrapped_result, (int)strlen(wrapped_result) + 1, MPI_CHAR, 0, TAG_RESULT, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
