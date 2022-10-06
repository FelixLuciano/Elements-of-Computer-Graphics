import os, shutil, time

import numba
import numpy as np


def main():
    t = 0
    A = 0

    os.system("title CodePen")

    while True:
        if t <= 1:
            # Smoothstep
            A = np.pi / 4 - t**2 * (3 - 2 * t) * np.pi / 2

        vectors = vertex_shader(A)
        frame = get_fragment(vectors.flatten("F"))

        t = (t + 0.03) % 1.5

        render(frame)
        clear()

        time.sleep(1 / 60)


canvas = tuple(shutil.get_terminal_size((80, 20)))
K2 = 32
K1 = np.min(canvas) * 8

@numba.njit
def vertex_shader(A):
    vertex = np.zeros(canvas)
    depth = np.zeros(canvas)
    cos_A = np.cos(A)
    sin_A = np.sin(A)
    cos_B = np.cos(np.pi / 3.1)
    sin_B = np.sin(np.pi / 3.1)

    for z in np.linspace(-0.5, 0.5, 32):
        for y in np.linspace(-1, 1, 32):
            for x in np.linspace(-1, 1, 32):
                if (
                    -0.75 <= x <= 0.75
                    and -0.75 <= y <= 0.75
                    or -0.75 <= x <= 0.75
                    and -0.4 <= z <= 0.4
                    or -0.75 <= y <= 0.75
                    and -0.4 <= z <= 0.4
                ):
                    # Make a hole in the middle
                    continue

                c_x = (cos_A * x + sin_A * y) * 2
                c_y = (-sin_A * x + cos_A * y) * cos_B + z * sin_B
                c_z = (cos_B * z + sin_B * (sin_A * x - cos_A * y)) + K2
                z_1 = 1 / c_z
                p_x = int(canvas[0] / 2 + K1 * c_x * z_1)
                p_y = int(canvas[1] / 2 - K1 * c_y * z_1)

                if z_1 > depth[p_x][p_y]:
                    vertex[p_x][p_y] = 1

    return vertex


@np.vectorize
def get_fragment(value):
    return " " if value == 0 else "#"


def render(frame):
    print("".join(frame), end="\r")


def clear(size=canvas[1]):
    print(f"\x1b[{size}A", end="\r")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
