#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
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


__global__ void initialize_u(my_type *u, int N, int M)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < N && y < M)
    {
        u[x + y * N] = 0.0;
    }
}

__global__ void define_c(my_type *c, my_type c_max, int N, int M)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < N && y < M)
    {
        c[x + y * N] = c_max;
    }
}

__global__ void define_initial_condition(my_type *u, int N, int M, int x0, int y0)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < N && y < M)
    {
        my_type a = 0.2;
        u[x + y * N] = exp(-a * ((x - x0) * (x - x0) + (y - y0) * (y - y0)));
    }
}

__global__ void calculate_wave(my_type *u_next, my_type *u_cur, my_type *u_prev, my_type *c, int N, int M, my_type dt, my_type dh)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x > 0 && x < N - 1 && y > 0 && y < M - 1)
    {
        my_type alpha = c[x + y * N] * c[x + y * N] * dt * dt / (dh * dh);
        u_next[x + y * N] = 2 * u_cur[x + y * N] - u_prev[x + y * N] +
                            alpha * (u_cur[(x + 1) + y * N] - 2 * u_cur[x + y * N] + u_cur[(x - 1) + y * N] +
                                     u_cur[x + (y + 1) * N] - 2 * u_cur[x + y * N] + u_cur[x + (y - 1) * N]);
    }
}

void ricker(my_type f0, my_type t0, my_type* time_arr, my_type* s_arr, int N_time)
{
    my_type arg;
    for (int i = 0; i < N_time; i++)
    {
        arg = M_PI * f0 * (time_arr[i] - t0);
        s_arr[i] = (2 * arg * arg - 1) * exp(-arg * arg);
    }
}



int main(int argc, char *argv[])
{
    
    int threads = atoi(argv[1]);
    int DIMENSION = atoi(argv[2]);
    sleep(2);
    const int N = DIMENSION; // Número de elementos espaciais na direção x
    const int M = DIMENSION; // Número de elementos espaciais na direção y
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

    for (int i = 0; i < N_time; i++)
    {
        time_arr[i] = i * dt;
    }
    my_type f0 = 100.0; // Frequência da onda de Ricker [Hz]
    my_type t0 = 0.01; // Deslocamento da onda de Ricker [s]
    ricker(f0, t0, time_arr, s_arr, N_time);

    my_type *d_u_next, *d_u_cur, *d_u_prev, *d_c;
    cudaMalloc((void**)&d_u_next, sizeof(my_type) * N * M);
    cudaMalloc((void**)&d_u_cur, sizeof(my_type) * N * M);
    cudaMalloc((void**)&d_u_prev, sizeof(my_type) * N * M);
    cudaMalloc((void**)&d_c, sizeof(my_type) * N * M);

    dim3 threadsPerBlock(threads, threads);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEvent_t initialize_u_start1, initialize_u_stop1, define_c_start, define_c_stop, define_initial_condition_start, define_initial_condition_stop, calculate_wave_start, calculate_wave_stop, initialize_u_start2, initialize_u_stop2, initialize_u_start3, initialize_u_stop3;
    cudaEventCreate(&initialize_u_start1);
    cudaEventCreate(&initialize_u_stop1);
    cudaEventCreate(&initialize_u_start2);
    cudaEventCreate(&initialize_u_stop2);
    cudaEventCreate(&initialize_u_start3);
    cudaEventCreate(&initialize_u_stop3);
    cudaEventCreate(&define_c_start);
    cudaEventCreate(&define_c_stop);
    cudaEventCreate(&define_initial_condition_start);
    cudaEventCreate(&define_initial_condition_stop);
    cudaEventCreate(&calculate_wave_start);
    cudaEventCreate(&calculate_wave_stop);

    print_time("inicio: ");
    cudaEventRecord(initialize_u_start1);
    initialize_u<<<numBlocks, threadsPerBlock>>>(d_u_cur, N, M);
    cudaEventRecord(initialize_u_stop1);

    cudaEventRecord(initialize_u_start2);
    initialize_u<<<numBlocks, threadsPerBlock>>>(d_u_prev, N, M);
    cudaEventRecord(initialize_u_stop2);
    
    cudaEventRecord(initialize_u_start3);
    initialize_u<<<numBlocks, threadsPerBlock>>>(d_u_next, N, M);
    cudaEventRecord(initialize_u_stop3);
    
    cudaEventRecord(define_c_start);
    define_c<<<numBlocks, threadsPerBlock>>>(d_c, c_max, N, M);
    cudaEventRecord(define_c_stop);
    
    cudaEventRecord(define_initial_condition_start); 
    define_initial_condition<<<numBlocks, threadsPerBlock>>>(d_u_cur, N, M, x0, y0);
    cudaEventRecord(define_initial_condition_stop);

    float elapsedTimeFor, total = 0;

    for (int i = 0; i < N_time; i++)
    {
        cudaEventRecord(calculate_wave_start);
        calculate_wave<<<numBlocks, threadsPerBlock>>>(d_u_next, d_u_cur, d_u_prev, d_c, N, M, dt, dh);
        cudaEventRecord(calculate_wave_stop);
        cudaEventSynchronize(calculate_wave_stop);
        cudaEventElapsedTime(&elapsedTimeFor, calculate_wave_start, calculate_wave_stop);
        total +=elapsedTimeFor;

    }
    total /= N_time;

    print_time("fim :");
    cudaEventSynchronize(define_initial_condition_stop);
    cudaEventSynchronize(define_c_stop);
    cudaEventSynchronize(initialize_u_stop3);
    cudaEventSynchronize(initialize_u_stop2);
    cudaEventSynchronize(initialize_u_stop1);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, define_initial_condition_start, define_initial_condition_stop);
    printf("define_initial_condition: %f ms\n", elapsedTime);
    cudaEventElapsedTime(&elapsedTime, define_c_start, define_c_stop);
    printf("define_c: %f ms\n", elapsedTime);
    cudaEventElapsedTime(&elapsedTime, initialize_u_start1, initialize_u_stop1); 
    printf("initialize_u1: %f ms\n", elapsedTime);
    cudaEventElapsedTime(&elapsedTime, initialize_u_start2, initialize_u_stop2);
    printf("initialize_u2: %f ms\n", elapsedTime);
    cudaEventElapsedTime(&elapsedTime, initialize_u_start3, initialize_u_stop3);
    printf("initialize_u3: %f ms\n", elapsedTime);
    printf("calculate_wave: %f ms\n", total);


    // Libera a memória
    cudaFree(d_u_next);
    cudaFree(d_u_cur);
    cudaFree(d_u_prev);
    cudaFree(d_c);
    free(time_arr);
    free(s_arr);
    free(h_u);


    return 0;
}
