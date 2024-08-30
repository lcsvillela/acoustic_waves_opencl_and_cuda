#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/cl.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>


typedef double my_type;

void print_time(const char* label) {
    struct timeval tv;
    gettimeofday(&tv, NULL);

    struct tm *gmt = gmtime(&tv.tv_sec);

    printf("%s: %04d-%02d-%02d %02d:%02d:%02d.%06ld UTC\n", label,
           gmt->tm_year + 1900, gmt->tm_mon + 1, gmt->tm_mday,
           gmt->tm_hour, gmt->tm_min, gmt->tm_sec, tv.tv_usec);
}


void check_error(cl_int err, const char *operation) {
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Erro durante %s: %d\n", operation, err);
        exit(EXIT_FAILURE);
    }
}

char *load_kernel_source(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Erro ao abrir o arquivo do kernel: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    fseek(file, 0, SEEK_END);
    size_t size = ftell(file);
    rewind(file);

    char *source = (char *)malloc(size + 1);
    fread(source, 1, size, file);
    source[size] = '\0';

    fclose(file);
    return source;
}

void ricker(my_type f0, my_type t0, my_type* time_arr, my_type* s_arr, int N_time) {
    my_type arg;
    for (int i = 0; i < N_time; i++) {
        arg = M_PI * f0 * (time_arr[i] - t0);
        s_arr[i] = (2 * arg * arg - 1) * exp(-arg * arg);
    }
}

int main(int argc, char *argv[]) {
    int threads = atoi(argv[1]);
    int DIMENSION = atoi(argv[2]);
    sleep(2);
    const int N = DIMENSION; // Número de elementos espaciais na direção x
    const int M = DIMENSION; 
    const my_type dh = 1; // Passo espacial [m]
    const my_type c_max = 3000.0; // Velocidade máxima [m/s]
    const my_type dt = (dh / c_max) * 0.1; // Passo temporal [s]
    const my_type time_duration = 8; // Duração total da simulação [s]
    const int N_time = (int)(time_duration / dt); // Número de elementos temporais
    const int x0 = N / 2; // Posição da fonte na direção x
    const int y0 = M / 2; // Posição da fonte na direção y

    my_type *time_arr = (my_type*) malloc(sizeof(my_type) * N_time);
    my_type *s_arr = (my_type*) malloc(sizeof(my_type) * N_time);
    my_type *h_u = (my_type*) malloc(sizeof(my_type) * N * M);

    for (int i = 0; i < N_time; i++) {
        time_arr[i] = i * dt;
    }
    my_type f0 = 100.0; // Frequência da onda de Ricker [Hz]
    my_type t0 = 0.01; // Deslocamento da onda de Ricker [s]
    ricker(f0, t0, time_arr, s_arr, N_time);

    const char *kernel_source = 
        "__kernel void initialize_u(__global double *u, int N, int M) {"
        "    int x = get_global_id(0);"
        "    int y = get_global_id(1);"
        "    if (x < N && y < M) {"
        "        u[x + y * N] = 0.0;"
        "    }"
        "}"
        "__kernel void define_c(__global double *c, double c_max, int N, int M) {"
        "    int x = get_global_id(0);"
        "    int y = get_global_id(1);"
        "    if (x < N && y < M) {"
        "        c[x + y * N] = c_max;"
        "    }"
        "}"
        "__kernel void define_initial_condition(__global double *u, int N, int M, int x0, int y0) {"
        "    int x = get_global_id(0);"
        "    int y = get_global_id(1);"
        "    if (x < N && y < M) {"
        "        double a = 0.2;"
        "        u[x + y * N] = exp(-a * ((x - x0) * (x - x0) + (y - y0) * (y - y0)));"
        "    }"
        "}"
        "__kernel void calculate_wave(__global double *u_next, __global double *u_cur, __global double *u_prev, __global double *c, int N, int M, double dt, double dh) {"
        "    int x = get_global_id(0);"
        "    int y = get_global_id(1);"
        "    if (x > 0 && x < N - 1 && y > 0 && y < M - 1) {"
        "        double alpha = c[x + y * N] * c[x + y * N] * dt * dt / (dh * dh);"
        "        u_next[x + y * N] = 2 * u_cur[x + y * N] - u_prev[x + y * N] +"
        "                            alpha * (u_cur[(x + 1) + y * N] - 2 * u_cur[x + y * N] + u_cur[(x - 1) + y * N] +"
        "                                     u_cur[x + (y + 1) * N] - 2 * u_cur[x + y * N] + u_cur[x + (y - 1) * N]);"
        "    }"
        "}";

    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_device_id devices[2];
    cl_uint num_devices;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel_init, kernel_c, kernel_cond, kernel_wave;
    cl_int err;

    cl_event calculate_wave_event, define_initial_condition_event, define_c_event, initialize_u_event1, initialize_u_event2, initialize_u_event3;
    cl_ulong calculate_wave_start, calculate_wave_stop, define_initial_condition_start, define_initial_condition_stop,  define_c_start, define_c_stop, initialize_u_start1, initialize_u_stop1, initialize_u_start2, initialize_u_stop2, initialize_u_start3, initialize_u_stop3;
    double total_time = 0, time;

    err = clGetPlatformIDs(1, &platform_id, NULL);
    check_error(err, "clGetPlatformIDs");

    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    printf("%d", err);
    check_error(err, "clGetDeviceIDs");

    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    check_error(err, "clCreateContext");




    queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
    check_error(err, "clCreateCommandQueue");

    program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, &err);
    check_error(err, "clCreateProgramWithSource");

    err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        // Obtém o log de build e exibe
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *)malloc(log_size);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "Erro durante a compilação do programa:\n%s\n", log);
        free(log);
        exit(EXIT_FAILURE);
    }

    kernel_init = clCreateKernel(program, "initialize_u", &err);
    check_error(err, "clCreateKernel initialize_u");

    kernel_c = clCreateKernel(program, "define_c", &err);
    check_error(err, "clCreateKernel define_c");

    kernel_cond = clCreateKernel(program, "define_initial_condition", &err);
    check_error(err, "clCreateKernel define_initial_condition");

    kernel_wave = clCreateKernel(program, "calculate_wave", &err);
    check_error(err, "clCreateKernel calculate_wave");

    cl_mem d_u_next = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(my_type) * N * M, NULL, &err);
    check_error(err, "clCreateBuffer d_u_next");

    cl_mem d_u_cur = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(my_type) * N * M, NULL, &err);
    check_error(err, "clCreateBuffer d_u_cur");

    cl_mem d_u_prev = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(my_type) * N * M, NULL, &err);
    check_error(err, "clCreateBuffer d_u_prev");

    cl_mem d_c = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(my_type) * N * M, NULL, &err);
    check_error(err, "clCreateBuffer d_c");

    size_t global_size[2] = {threads, threads};
    size_t local_size[2] = {(N+threads-1)/threads, (M+threads-1)/M};

    print_time("inicio ");
    err = clSetKernelArg(kernel_init, 0, sizeof(cl_mem), &d_u_cur);
    err |= clSetKernelArg(kernel_init, 1, sizeof(int), &N);
    err |= clSetKernelArg(kernel_init, 2, sizeof(int), &M);
    check_error(err, "clSetKernelArg initialize_u");

    err = clEnqueueNDRangeKernel(queue, kernel_init, 2, NULL, global_size, local_size, 0, NULL, &initialize_u_event1);
    check_error(err, "clEnqueueNDRangeKernel initialize_u");
    clWaitForEvents(1, &initialize_u_event1);
    clFinish(queue);
    clGetEventProfilingInfo(initialize_u_event1, CL_PROFILING_COMMAND_START, sizeof(initialize_u_start1), &initialize_u_start1, NULL);
    clGetEventProfilingInfo(initialize_u_event1, CL_PROFILING_COMMAND_END, sizeof(initialize_u_stop1), &initialize_u_stop1, NULL);

    time= initialize_u_stop1-initialize_u_start1;
    printf("initialize_u1: %lf\n", time);


    err = clSetKernelArg(kernel_init, 0, sizeof(cl_mem), &d_u_prev);
    check_error(err, "clSetKernelArg initialize_u prev");

    err = clEnqueueNDRangeKernel(queue, kernel_init, 2, NULL, global_size, local_size, 0, NULL, &initialize_u_event2);
    check_error(err, "clEnqueueNDRangeKernel initialize_u prev");
    clWaitForEvents(1, &initialize_u_event2);
    clFinish(queue);
    clGetEventProfilingInfo(initialize_u_event2, CL_PROFILING_COMMAND_START, sizeof(initialize_u_start2), &initialize_u_start2, NULL);
    clGetEventProfilingInfo(initialize_u_event2, CL_PROFILING_COMMAND_END, sizeof(initialize_u_stop2), &initialize_u_stop2, NULL);

    time = initialize_u_stop2-initialize_u_start2;
    printf("initialize_u2: %lf\n", time);


    err = clSetKernelArg(kernel_init, 0, sizeof(cl_mem), &d_u_next);
    check_error(err, "clSetKernelArg initialize_u next");

    err = clEnqueueNDRangeKernel(queue, kernel_init, 2, NULL, global_size, local_size, 0, NULL, &initialize_u_event3);
    check_error(err, "clEnqueueNDRangeKernel initialize_u next");
    clWaitForEvents(1, &initialize_u_event3);
    clFinish(queue);
    clGetEventProfilingInfo(initialize_u_event3, CL_PROFILING_COMMAND_START, sizeof(initialize_u_start3), &initialize_u_start3, NULL);
    clGetEventProfilingInfo(initialize_u_event3, CL_PROFILING_COMMAND_END, sizeof(initialize_u_stop3), &initialize_u_stop3, NULL);

    time = initialize_u_stop3-initialize_u_start3;
    printf("initialize_u3: %lf\n", time);

    err = clSetKernelArg(kernel_c, 0, sizeof(cl_mem), &d_c);
    err |= clSetKernelArg(kernel_c, 1, sizeof(my_type), &c_max);
    err |= clSetKernelArg(kernel_c, 2, sizeof(int), &N);
    err |= clSetKernelArg(kernel_c, 3, sizeof(int), &M);
    check_error(err, "clSetKernelArg define_c");


    err = clEnqueueNDRangeKernel(queue, kernel_c, 2, NULL, global_size, local_size, 0, NULL, &define_c_event);
    check_error(err, "clEnqueueNDRangeKernel define_c");

    clWaitForEvents(1, &define_c_event);
    clFinish(queue);
    clGetEventProfilingInfo(define_c_event, CL_PROFILING_COMMAND_START, sizeof(define_c_start), &define_c_start, NULL);
    clGetEventProfilingInfo(define_c_event, CL_PROFILING_COMMAND_END, sizeof(define_c_stop), &define_c_stop, NULL);

    time = define_c_stop-define_c_start;
    printf("define_c: %lf\n", time);

    err = clSetKernelArg(kernel_cond, 0, sizeof(cl_mem), &d_u_cur);
    err |= clSetKernelArg(kernel_cond, 1, sizeof(int), &N);
    err |= clSetKernelArg(kernel_cond, 2, sizeof(int), &M);
    err |= clSetKernelArg(kernel_cond, 3, sizeof(int), &x0);
    err |= clSetKernelArg(kernel_cond, 4, sizeof(int), &y0);
    check_error(err, "clSetKernelArg define_initial_condition");

    err = clEnqueueNDRangeKernel(queue, kernel_cond, 2, NULL, global_size, local_size, 0, NULL, &define_initial_condition_event);
    check_error(err, "clEnqueueNDRangeKernel define_initial_condition");
    clWaitForEvents(1, &define_initial_condition_event);
    clFinish(queue);
    clGetEventProfilingInfo(define_initial_condition_event, CL_PROFILING_COMMAND_START, sizeof(define_initial_condition_start), &define_initial_condition_start, NULL);
    clGetEventProfilingInfo(define_initial_condition_event, CL_PROFILING_COMMAND_END, sizeof(define_initial_condition_stop), &define_initial_condition_stop, NULL);

    time = define_initial_condition_stop-define_initial_condition_start;
    printf("define_initial_condition: %lf\n", time);

    print_time("inicio for: ");
    for (int i = 0; i < N_time; i++) {
        err = clSetKernelArg(kernel_wave, 0, sizeof(cl_mem), &d_u_next);
        err |= clSetKernelArg(kernel_wave, 1, sizeof(cl_mem), &d_u_cur);
        err |= clSetKernelArg(kernel_wave, 2, sizeof(cl_mem), &d_u_prev);
        err |= clSetKernelArg(kernel_wave, 3, sizeof(cl_mem), &d_c);
        err |= clSetKernelArg(kernel_wave, 4, sizeof(int), &N);
        err |= clSetKernelArg(kernel_wave, 5, sizeof(int), &M);
        err |= clSetKernelArg(kernel_wave, 6, sizeof(my_type), &dt);
        err |= clSetKernelArg(kernel_wave, 7, sizeof(my_type), &dh);
       check_error(err, "clSetKernelArg calculate_wave");

        err = clEnqueueNDRangeKernel(queue, kernel_wave, 2, NULL, global_size, local_size, 0, NULL, &calculate_wave_event);
        check_error(err, "clEnqueueNDRangeKernel calculate_wave");
    clWaitForEvents(1, &calculate_wave_event);
    clFinish(queue);
    clGetEventProfilingInfo(calculate_wave_event, CL_PROFILING_COMMAND_START, sizeof(calculate_wave_start), &calculate_wave_start, NULL);
    clGetEventProfilingInfo(calculate_wave_event, CL_PROFILING_COMMAND_END, sizeof(calculate_wave_stop), &calculate_wave_stop, NULL);

    total_time += calculate_wave_stop-calculate_wave_start;

        //if (i % 100 == 0) {
        //    clEnqueueReadBuffer(queue, d_u_cur, CL_TRUE, 0, sizeof(my_type) * N * M, h_u, 0, NULL, NULL);

          //  char filename[50];
           // snprintf(filename, sizeof(filename), "wave_%04d.txt", i);
           // FILE *file = fopen(filename, "w");
           // if (file == NULL) {
           //     printf("Erro ao abrir o arquivo para escrita\n");
           //     exit(-1);
           // }
           // for (int y = 0; y < M; y++) {
           //     for (int x = 0; x < N; x++) {
           //         fprintf(file, "%f ", h_u[x + y * N]);
           //     }
           //     fprintf(file, "\n");
           // }
           // fclose(file);
        //}

        cl_mem temp = d_u_prev;
        d_u_prev = d_u_cur;
        d_u_cur = d_u_next;
        d_u_next = temp;
    }
    print_time("final ");

    total_time /= N_time;
    printf("calculate_wave: %f\n", total_time);

    clReleaseMemObject(d_u_next);
    clReleaseMemObject(d_u_cur);
    clReleaseMemObject(d_u_prev);
    clReleaseMemObject(d_c);
    clReleaseKernel(kernel_init);
    clReleaseKernel(kernel_c);
    clReleaseKernel(kernel_cond);
    clReleaseKernel(kernel_wave);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(time_arr);
    free(s_arr);
    free(h_u);

    return 0;
}
