import math
import time

A = 0.0
B = math.pi
N = 50_000_000  # numero de rectangulos


def f(x):
    return math.sin(x)


def riemann_secuencial(a, b, n):
    dx = (b - a) / n
    suma = 0.0
    for i in range(n):
        x = a + (i + 0.5) * dx  # punto medio del rectangulo i
        suma += f(x) * dx       # area del rectangulo i
    return suma


if __name__ == "__main__":
    inicio = time.perf_counter()
    resultado = riemann_secuencial(A, B, N)
    fin = time.perf_counter()

    print("Rectangulos:", N)
    print("Integral aproximada:", resultado)
    print("Valor exacto:", 2.0, "| error:", abs(resultado - 2.0))
    print("Tiempo CPU:", fin - inicio, "segundos")
