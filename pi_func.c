#include <stdio.h>
#include <omp.h>
 
#pragma omp declare simd uniform(delta_x)
double integracao(int i, double delta_x){
    double x = (i + 0.5) * delta_x;
    return 4.0 / (1.0 + x * x);
}
 
int main(){
    int n = 1000000;
    double delta_x = 1.0 / n;
    double soma = 0.0;
 
    #pragma omp simd reduction(+:soma)
    for (int i = 0; i < n; i++){
        soma += integracao(i, delta_x);
    }
 
    double pi = soma * delta_x;
    printf("pi estimado = %.15f\n", pi);
 
    return 0;
}
