#include <stdio.h>
#include <omp.h>

double f(double x){
    return (4/(1+x*x));
}

int main() {
    double a = 1;
    double b = 0;
    int n = 100000;  
    double h = 0;  				        
    h = (b-a)/n;
    double area = 0;

    int num_threads = omp_get_max_threads();
    double somas_parciais[num_threads];

    for (int i = 0; i < num_threads; i++) {
        somas_parciais[i] = 0;
    }

    #pragma omp parallel
    {
	    int tid = omp_get_thread_num();

        for (int i = tid; i < n; i += num_threads) {
            somas_parciais[tid] += f(a + i * h) * h;
        }
    }
    
    for (int i = 0; i < num_threads; i++) {
        area += somas_parciais[i];
    }

    printf("Resultado: %f\n", area);
    return 0;
}