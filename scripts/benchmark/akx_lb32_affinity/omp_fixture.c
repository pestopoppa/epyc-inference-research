#include <stdio.h>
#include <unistd.h>
#include <omp.h>
int main(void) {
#pragma omp parallel num_threads(8)
 {
#pragma omp barrier
#pragma omp single
  { printf("ready %d\n", omp_get_num_threads()); fflush(stdout); getchar(); }
#pragma omp barrier
 }
 return 0;
}
