from multiprocessing import Pool, cpu_count
import timeit


def g(x):
    return x**3 + 2


def f(x):
    return x**2, g(x)

if __name__ == "__main__":
  inputs = [i for i in range(32)]
  num_cores = cpu_count()
  p = Pool(num_cores)
  t1 = timeit.Timer()
  result = p.map(f, inputs)
  print(t1.timeit(1))
  print(result)