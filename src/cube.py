import os, shutil, time

import numba
import numpy as np


def main():
    A = np.pi / 4
    B = -np.pi / 4

    os.system("title Cube³")

    while True:
        vectors = vertex_shader(A, B)
        frame = get_fragment(vectors.flatten("F"))

        A = (A + 0.05) % (np.pi * 2)
        B = (B - 0.02) % (np.pi * 2)

        render(frame)
        clear()

        time.sleep(1 / 60)


canvas = tuple(shutil.get_terminal_size((80, 20)))
K2 = 32
K1 = np.min(canvas) * 6

@numba.njit
def vertex_shader(A, B):
    vertex = np.zeros(canvas)
    depth = np.zeros(canvas)
    cos_A = np.cos(A)
    sin_A = np.sin(A)
    cos_B = np.cos(B)
    sin_B = np.sin(B)

    for z in np.linspace(-1, 1, 32):
        for y in np.linspace(-1, 1, 32):
            if z == y:
                # Ignore edges
                continue

            for x in np.linspace(-1, 1, 32):
                if (
                    -0.66 <= x <= 0.66
                    and -0.66 <= y <= 0.66
                    or -0.66 <= x <= 0.66
                    and -0.66 <= z <= 0.66
                    or -0.66 <= y <= 0.66
                    and -0.66 <= z <= 0.66
                    or x == y
                    or x == z
                ):
                    # Make a hole in the middle and ignore edges
                    continue

                x_abs = np.abs(x)
                y_abs = np.abs(y)
                z_abs = np.abs(z)
                c_x = (cos_A * x + sin_A * y) * 2
                c_y = (-sin_A * x + cos_A * y) * cos_B + z * sin_B
                c_z = (cos_B * z + sin_B * (sin_A * x - cos_A * y)) + K2
                z_1 = 1 / c_z
                p_x = int(canvas[0] / 2 + K1 * c_x * z_1)
                p_y = int(canvas[1] / 2 - K1 * c_y * z_1)
                N_x = 0
                N_y = 0
                N_z = 0

                if x_abs > y_abs and x_abs > z_abs:
                    N_x = np.sign(x)

                if y_abs > x_abs and y_abs > z_abs:
                    N_y = np.sign(y)

                if z_abs > x_abs and z_abs > y_abs:
                    N_z = np.sign(z)

                L = np.abs(
                    (cos_A * N_x + sin_A * N_y)
                    + ((-sin_A * N_x + cos_A * N_y) * cos_B + N_z * sin_B)
                    - (cos_B * N_z + sin_B * (sin_A * N_x - cos_A * N_y))
                )

                if L > 0 and z_1 > depth[p_x][p_y]:
                    vertex[p_x][p_y] = L / np.sqrt(3)
                    depth[p_x][p_y] = z_1

    return vertex


@np.vectorize
def get_fragment(value):
    if value == 0:
        return " "

    i = min(int(value * 12), 11)

    return ".,-~:;=!*#$@"[i]


def render(frame):
    print("".join(frame), end="\r")


def clear(size=canvas[1]):
    print(f"\x1b[{size}A", end="\r")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
