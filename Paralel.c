#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

/* Headers */
double Trap(double a, double b, int n);
double f(double x);

int main(int argc, char *argv[]) {
  /* Variable declaration */
  double global_result = 0.0;
  double a, b;
  int n, thread_count;

  /* Input */
  // Number of Threads
  thread_count = atoi(argv[1]);

  // Input Parameters
  printf("Enter a, b and n\n");
  scanf("%lf %lf %d", &a, &b, &n);

#pragma omp parallel num_threads(thread_count)
  {
    double my_result = 0.0;
    my_result += Trap(a, b, n);

#pragma omp critical
    global_result += my_result;
  }
  printf("Area under the curve from %lf to %lf with %d trapezoids is %lf", a, b,
         n, global_result);

  return 0;
}

double Trap(double a, double b, int n) {
  int i, local_n;
  double local_a, local_b;
  double h, local_partial_sum;

  int my_rank = omp_get_thread_num();
  int thread_count = omp_get_num_threads();

  // 1. Length of the trapezoid bases (h)
  h = (b - a) / n;

  // 2. Number of trapezoids assigned to each thread
  local_n = n / thread_count;

  // 3. Identify left (local_a) and right (local_b) endpoints
  local_a = a + my_rank * local_n * h;
  local_b = local_a + local_n * h;

  // 4. Compute local_partial_sum
  local_partial_sum = (f(local_a) + f(local_b)) / 2;

  for (i = 1; i <= local_n - 1; i++) {
    local_partial_sum += f(local_a + i * h);
  }
  return local_partial_sum * h;
}

double f(double x) { return x * x; }
