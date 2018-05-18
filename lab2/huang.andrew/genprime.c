#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <time.h>
/* Author: Andrew Huang
   Parallel Computing Spring 2018*/
/*functions*/
typedef enum { false, true } bool; // booltype
void Hello(void);
int read(char * argv[]);
void clean(char *outputf);
void sieve(int threads);
void output(char * outputf);
/* globals */
int *values; // original array
int *primes; //prime values
int primes_s = 1; //pos of current prime
int N;
bool *primeflag; // marks the ith position if its prime


int read(char * argv[]){
    N = atoi(argv[1]);
    values = (int * ) malloc(N * sizeof(int));
    if (!values){
        printf("Could not init values array");
        exit(1);
    }
    primeflag = (bool * ) malloc(N * sizeof(bool));
    if (!primeflag){
        printf("could not init bool array (primes)");
        exit(1);
    }
    int i;
    for (i = 0; i < N; i++){
        values[i] = i;
        primeflag[i] = true;
    }
    int fl = ((N+1)/2) + 1;
    primes = (int *) malloc( fl * sizeof(int));
    if (!primes){
        printf("could not init prime array");
        exit(1);
    }
    primes[0] = 2;
    return atoi(argv[2]);
}
void sieve(int threads){
    int i;
    for (i = 2; i < N; i ++){
        if (primeflag[i]){
            #pragma omp parallel num_threads(threads)
            {
                int j;
                #pragma omp parallel for
                for (j = i; j < N; j += i){
                    primeflag[j] = false;
                }
            }
            primes[primes_s] = i;
            primes_s++;
        }
    }
}
void output(char * outputf){
    FILE *out = fopen(outputf, "w");
    if(out == NULL){
        printf("File could not be created");
        exit(1);
    }
    int i;
    for (i = 1; i < primes_s; i++){
        fprintf(out, "%d, %d, %d\n", i, primes[i], primes[i]-primes[i-1]);
    }
    fclose(out);
}
void clean(char *outputf){
    free(outputf);
    free(values);
    free(primes);
    free(primeflag);
}
void Hello(void){
    int my_rank = omp_get_thread_num();
    int thread_count = omp_get_num_threads();
    printf("hello from thread %d of %d\n", my_rank, thread_count);
}
int main(int argc, char* argv[]){
    // int thread_count = strtol(argv[1], NULL, 10);
    // # pragma omp parallel num_threads(thread_count)
    // Hello();
    double tstart, ttaken;
    if (argc != 3){
        printf("Usage: ./genprime <N_value> <number_of_threads>\n");
        exit(1);
    }
    int threads = read(argv); // threads is now set;
    char * ext = ".txt\0";
    char * outputf = malloc(strlen(argv[1]) + strlen(ext)+1);
    outputf[0] = '\0';
    strcat(outputf, argv[1]);
    strcat(outputf, ext);

    tstart = omp_get_wtime();
    sieve(threads);

    int rank = omp_get_thread_num();
    if (rank == 0 ){
        ttaken = omp_get_wtime() - tstart;
        printf("Time take for the main part: %f\n", ttaken);
        output(outputf);
    }
    clean(outputf);
     // cleanup
    return 0;
}
