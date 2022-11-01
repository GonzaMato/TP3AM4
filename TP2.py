import math

# y' = z
# y'' = z' = w
# y'''= w' = e^x - 2w + z + 2y
e = math.e
f1 = lambda x, y, z, w: z
f2 = lambda x, y, z, w: w
f3 = lambda x, y, z, w: e ** x - 2 * w + z + 2 * y
solución = lambda x: (-11/36) * e ** x + (-4/9) * e ** (-2 * x) + 1.75 * e ** (-x) + (1 / 6) * x * e ** x


# Runge-Kutta de orden 4
def RK_orden4(funcion, x0, y0, h, n):
    y = [y0]
    x = [x0]
    for i in range(1, n):
        x.append(x0 + i * h)
        y.append(0)
    for i in range(n):
        k1 = h * funcion(x[i], y[i])
        k2 = h * funcion(x[i] + (h / 2), y[i] + (k1 / 2))
        k3 = h * funcion(x[i] + (h / 2), y[i] + (k2 / 2))
        k4 = h * funcion(x[i] + h, y[i] + k3)
        y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return x, y


# Se realiza primero Runge Kutta para sistema de ecuaciones. Acoplado con el método predictor-corrector AB y AM,
# se debe hacer RK hasta y4 (4ta iteración de i), luego seguir con AB y AM a partir de y5, hasta llegar a y(b)
def resolution_3_EDO_system(f1, f2, f3, x0, y0, z0, w0, h, b):
    y = [y0]
    x = [x0]
    z = [z0]
    w = [w0]
    for i in range(1, 5):
        x.append(x0 + i * h)
        y.append(0)
        z.append(0)
        w.append(0)
    for i in range(4):
        k1 = h * f1(x[i], y[i], z[i], w[i])  # z
        l1 = h * f2(x[i], y[i], z[i], w[i])  # w
        m1 = h * f3(x[i], y[i], z[i], w[i])  # w'

        k2 = h * f1(x[i] + 0.5 * h, y[i] + 0.5 * k1, z[i] + 0.5 * l1, w[i] + 0.5 * m1)
        l2 = h * f2(x[i] + 0.5 * h, y[i] + 0.5 * k1, z[i] + 0.5 * l1, w[i] + 0.5 * m1)
        m2 = h * f3(x[i] + 0.5 * h, y[i] + 0.5 * k1, z[i] + 0.5 * l1, w[i] + 0.5 * m1)

        k3 = h * f1(x[i] + 0.5 * h, y[i] + 0.5 * k2, z[i] + 0.5 * l2, w[i] + 0.5 * m2)
        l3 = h * f2(x[i] + 0.5 * h, y[i] + 0.5 * k2, z[i] + 0.5 * l2, w[i] + 0.5 * m2)
        m3 = h * f3(x[i] + 0.5 * h, y[i] + 0.5 * k2, z[i] + 0.5 * l2, w[i] + 0.5 * m2)

        k4 = h * f1(x[i] + h, y[i] + k3, z[i] + l3, w[i] + m3)
        l4 = h * f2(x[i] + h, y[i] + k3, z[i] + l3, w[i] + m3)
        m4 = h * f3(x[i] + h, y[i] + k3, z[i] + l3, w[i] + m3)

        y[i + 1] = y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        z[i + 1] = z[i + 1] = z[i] + (l1 + 2 * l2 + 2 * l3 + l4) / 6
        w[i + 1] = w[i + 1] = w[i] + (m1 + 2 * m2 + 2 * m3 + m4) / 6
    # Adams-Bashforth de 5 pasos
    i = 4
    while x[i] != b:
        x.append(x0 + (i + 1) * h)
        y.append(y[i] + (h / 720) * (1901 * f1(x[i], y[i], z[i], w[i]) -
                                     2774 * f1(x[i - 1], y[i - 1], z[i - 1], w[i - 1]) +
                                     2616 * f1(x[i - 2], y[i - 2], z[i - 2], w[i - 2]) -
                                     1274 * f1(x[i - 3], y[i - 3], z[i - 3], w[i - 3])) +
                                     251 * f1(x[i - 4], y[i - 4], z[i - 4], w[i - 4]))
        z.append(z[i] + (h / 720) * (1901 * f2(x[i], y[i], z[i], w[i]) -
                                     2774 * f2(x[i - 1], y[i - 1], z[i - 1], w[i - 1]) +
                                     2616 * f2(x[i - 2], y[i - 2], z[i - 2], w[i - 2]) -
                                     1274 * f2(x[i - 3], y[i - 3], z[i - 3], w[i - 3])) +
                                     251 * f2(x[i - 4], y[i - 4], z[i - 4], w[i - 4]))
        w.append(w[i] + (h / 720) * (1901 * f3(x[i], y[i], z[i], w[i]) -
                                     2774 * f3(x[i - 1], y[i - 1], z[i - 1], w[i - 1]) +
                                     2616 * f3(x[i - 2], y[i - 2], z[i - 2], w[i - 2]) -
                                     1274 * f3(x[i - 3], y[i - 3], z[i - 3], w[i - 3])) +
                                     251 * f3(x[i - 4], y[i - 4], z[i - 4], w[i - 4]))
        # Adams-Moulton de 5 pasos
        for j in range(10):
            y[i + 1] = y[i] + (h / 720) * (251 * f1(x[i + 1], y[i + 1], z[i + 1], w[i + 1]) +
                                           646 * f1(x[i], y[i], z[i], w[i]) -
                                           264 * f1(x[i - 1], y[i - 1], z[i - 1], w[i - 1]) +
                                           106 * f1(x[i - 2], y[i - 2], z[i - 2], w[i - 2]) -
                                           19 * f1(x[i - 3], y[i - 3], z[i - 3], w[i - 3]))
            z[i + 1] = z[i] + (h / 720) * (251 * f2(x[i + 1], y[i + 1], z[i + 1], w[i + 1]) +
                                           646 * f2(x[i], y[i], z[i], w[i]) -
                                           264 * f2(x[i - 1], y[i - 1], z[i - 1], w[i - 1]) +
                                           106 * f2(x[i - 2], y[i - 2], z[i - 2], w[i - 2]) -
                                           19 * f2(x[i - 3], y[i - 3], z[i - 3], w[i - 3]))
            w[i + 1] = w[i] + (h / 720) * (251 * f3(x[i + 1], y[i + 1], z[i + 1], w[i + 1]) +
                                           646 * f3(x[i], y[i], z[i], w[i]) -
                                           264 * f3(x[i - 1], y[i - 1], z[i - 1], w[i - 1]) +
                                           106 * f3(x[i - 2], y[i - 2], z[i - 2], w[i - 2]) -
                                           19 * f3(x[i - 3], y[i - 3], z[i - 3], w[i - 3]))
        i += 1
    return y[i]


# condiciones iniciales
x0 = 0
y0 = 1
z0 = -1
w0 = 0

# x=1 h=0,1 ; x=1 h= 0,05 ; x= 2 h= 0,1 ; x=2 h=0,05 ; x=3 h=0,5 ; x=3 h=0,1
h = [0.5, 0.1, 0.05]
x = [1, 2, 3]
print("X = 1 ; H = 0,1\nresultado mediante el método = " +
      str(resolution_3_EDO_system(f1, f2, f3, x0, y0, z0, w0, h[1], x[0])) +
      "\t Solución exacta = " + str(solución(x[0])) +
      "\nX = 1 ; H = 0,05\nresultado mediante el método = " +
      str(resolution_3_EDO_system(f1, f2, f3, x0, y0, z0, w0, h[2], x[0])) +
      "\t Solución exacta = " + str(solución(x[0])) +
      "\nX = 2 ; H = 0,1\nresultado mediante el método = " +
      str(resolution_3_EDO_system(f1, f2, f3, x0, y0, z0, w0, h[1], x[1])) +
      "\t Solución exacta = " + str(solución(x[1])) +
      "\nX = 2 ; H = 0,05\nresultado mediante el método = " +
      str(resolution_3_EDO_system(f1, f2, f3, x0, y0, z0, w0, h[2], x[1])) +
      "\t Solución exacta = " + str(solución(x[1])) +
      "\nX = 3 ; H = 0,5\nresultado mediante el método = " +
      str(resolution_3_EDO_system(f1, f2, f3, x0, y0, z0, w0, h[0], x[2])) +
      "\t Solución exacta = " + str(solución(x[2])) +
      "\nX = 3 ; H = 0,1\nresultado mediante el método = " +
      str(resolution_3_EDO_system(f1, f2, f3, x0, y0, z0, w0, h[1], x[2])) +
      "\t Solución exacta = " + str(solución(x[2])))
