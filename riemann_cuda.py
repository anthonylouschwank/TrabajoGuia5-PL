import math
import time

import numpy as np
from numba import cuda, float64

from riemann_secuencial import A, B, N, riemann_secuencial

HILOS_POR_BLOQUE = 256


@cuda.jit
def kernel_riemann(a, dx, n, parciales):
    # Memoria compartida del bloque para la reduccion
    compartida = cuda.shared.array(HILOS_POR_BLOQUE, dtype=float64)

    i = cuda.grid(1)
    tid = cuda.threadIdx.x

    area = 0.0
    if i < n:
        x = a + (i + 0.5) * dx          # punto medio del rectangulo i
        area = math.sin(x) * dx         # area del rectangulo i
    compartida[tid] = area
    cuda.syncthreads()

    # Reduccion en arbol: suma de las areas del bloque
    paso = HILOS_POR_BLOQUE // 2
    while paso > 0:
        if tid < paso:
            compartida[tid] += compartida[tid + paso]
        cuda.syncthreads()
        paso //= 2

    if tid == 0:
        parciales[cuda.blockIdx.x] = compartida[0]  # suma parcial del bloque


if __name__ == "__main__":
    # Tiempo CPU de referencia
    inicio = time.perf_counter()
    resultado_cpu = riemann_secuencial(A, B, N)
    t_cpu = time.perf_counter() - inicio

    dx = (B - A) / N
    bloques = (N + HILOS_POR_BLOQUE - 1) // HILOS_POR_BLOQUE

    # Calentamiento: compila el kernel (JIT) fuera de la medicion
    d_w = cuda.to_device(np.zeros(1, dtype=np.float64))
    kernel_riemann[1, HILOS_POR_BLOQUE](A, dx, 1, d_w)
    cuda.synchronize()

    # 1. Host -> Device (el arreglo de sumas parciales; a, dx y n son escalares)
    inicio = time.perf_counter()
    d_parciales = cuda.to_device(np.zeros(bloques, dtype=np.float64))
    cuda.synchronize()
    t_h2d = time.perf_counter() - inicio

    # 2. Kernel
    inicio = time.perf_counter()
    kernel_riemann[bloques, HILOS_POR_BLOQUE](A, dx, N, d_parciales)
    cuda.synchronize()
    t_kernel = time.perf_counter() - inicio

    # 3. Device -> Host
    inicio = time.perf_counter()
    parciales = d_parciales.copy_to_host()
    t_d2h = time.perf_counter() - inicio

    # 4. Suma final de las sumas parciales en el Host
    inicio = time.perf_counter()
    resultado_gpu = float(parciales.sum())
    t_suma = time.perf_counter() - inicio

    t_total = t_h2d + t_kernel + t_d2h + t_suma
    print("Rectangulos:", N, "| bloques:", bloques, "| hilos por bloque:", HILOS_POR_BLOQUE)
    print(f"Tiempo CPU:         {t_cpu:.6f} s")
    print(f"Transferencia H->D: {t_h2d:.6f} s")
    print(f"Kernel:             {t_kernel:.6f} s")
    print(f"Transferencia D->H: {t_d2h:.6f} s")
    print(f"Suma final (Host):  {t_suma:.6f} s")
    print(f"Total CUDA:         {t_total:.6f} s")
    print(f"Speedup (CPU/total CUDA): {t_cpu / t_total:.2f}x")
    print(f"Speedup (CPU/kernel):     {t_cpu / t_kernel:.2f}x")
    print("Integral CPU:", resultado_cpu)
    print("Integral GPU:", resultado_gpu)
    print("Diferencia CPU-GPU:", abs(resultado_cpu - resultado_gpu))
    print("Error vs exacto (2.0) GPU:", abs(resultado_gpu - 2.0))
    print("Resultado correcto:", math.isclose(resultado_cpu, resultado_gpu, rel_tol=1e-9))
