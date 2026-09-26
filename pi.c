#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define SEED 123456789UL

int main(){
    long long n;
    double inicio, fim;

    printf("n = ");
    scanf("%lld", &n);

    double delta_x = 1.0 / (double)n;
    double soma = 0.0;
    double pi_integracao;

    #pragma omp target teams distribute parallel for \
        map(to: delta_x, n) reduction(+:soma)
    for (long long i = 0; i < n; i++) {
        double x = (i + 0.5) * delta_x;
        soma += 4.0 / (1.0 + x * x);
    }

    pi_integracao = soma * delta_x;

    printf("\nintegracao numerica\n");
    printf("pi = %.15f\n", pi_integracao);

    long long pontos_dentro = 0;
    double pi_montecarlo;

    #pragma omp target teams distribute parallel for \
        map(to: n) reduction(+:pontos_dentro)
    for (long long i = 0; i < n; i++) {
        unsigned int s1 = (unsigned int)(SEED + i * 1103515245u + 12345u);
        unsigned int s2 = s1 * 1103515245u + 12345u;

        double x = (double)(s1 & 0x7fffffffU) / 2147483647.0;
        double y = (double)(s2 & 0x7fffffffU) / 2147483647.0;

        if (x * x + y * y <= 1.0)
            pontos_dentro++;
    }

    pi_montecarlo = 4.0 * (double)pontos_dentro / (double)n;

    printf("\nmontecarlo\n");
    printf("pi = %.15f\n", pi_montecarlo);

    return 0;
}