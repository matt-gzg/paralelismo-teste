#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define M 2147483647LL
#define A 48271LL
#define C 0LL
#define NUM_THREADS 8

long long potencia_mod(long long base, long long exp, long long mod){
    long long resultado = 1;
    base %= mod;
    while (exp > 0) {
        if (exp & 1) resultado = resultado * base % mod;
        base = base * base % mod;
        exp >>= 1;
    }
    return resultado;
}

int main(){
    long int n;
    double inicio, fim;

    printf("\nn = ");
    scanf("%ld", &n);

    omp_set_num_threads(NUM_THREADS);

    long long salto = potencia_mod(A, 2 * NUM_THREADS, M);
    long int contagem_lf = 0;

    inicio = omp_get_wtime();

    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        long long x = potencia_mod(A, 2 * id + 1, M);
        long int contagem_local = 0;

        #pragma omp for schedule(static)
        for (long int i = 0; i < n; i++) {
            long double u = (long double)x / M;
            x = (A * x + C) % M;
            long double v = (long double)x / M;
            x = salto * x % M;
            if (u * u + v * v <= 1.0L)
                contagem_local++;
        }

        #pragma omp atomic
        contagem_lf += contagem_local;
    }

    fim = omp_get_wtime();
    printf("\n[Leapfrog]\n");
    printf("Pi aprox. = %.9Lf\n", (long double)contagem_lf / n * 4);
    printf("Tempo     = %f s\n", fim - inicio);

    long int contagem_mlf = 0;
    inicio = omp_get_wtime();

    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        long long x = potencia_mod(A, (long long)id * (n / NUM_THREADS) * 2 + 1, M);
        long int contagem_local = 0;

        #pragma omp for schedule(static)
        for (long int i = 0; i < n; i++) {
            long double u = (long double)x / M;
            x = (A * x + C) % M;
            long double v = (long double)x / M;
            x = (A * x + C) % M;
            if (u * u + v * v <= 1.0L)
                contagem_local++;
        }

        #pragma omp atomic
        contagem_mlf += contagem_local;
    }

    fim = omp_get_wtime();
    printf("\n[Leapfrog Modificado]\n");
    printf("Pi aprox. = %.9Lf\n", (long double)contagem_mlf / n * 4);
    printf("Tempo     = %f s\n", fim - inicio);

    return 0;
}