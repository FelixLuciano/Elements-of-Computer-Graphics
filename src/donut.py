import os, shutil, time

import numba
import numpy as np


def main():
    A = np.pi / 2
    B = np.pi / 2

    os.system("title Donut")

    while True:
        vectors = vertex_shader(A, B)
        frame = get_fragment(vectors.flatten("F"))

        A = (A + 0.07) % (np.pi * 2)
        B = (B + 0.03) % (np.pi * 2)

        render(frame)
        clear()

        time.sleep(1 / 60)


canvas = tuple(shutil.get_terminal_size((80, 20)))
K2 = 64
K1 = np.min(canvas) * 6

@numba.njit
def vertex_shader(A, B):
    """Inspired by Andy Sloane's Donut math
    From https://www.a1k0n.net/2011/07/20/donut-math.html
    """
    vertex = np.zeros(canvas)
    depth = np.zeros(canvas)
    cos_A = np.cos(A)
    sin_A = np.sin(A)
    cos_B = np.cos(B)
    sin_B = np.sin(B)

    for theta in np.linspace(0, 2 * np.pi, 128):
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        for phi in np.linspace(0, 2 * np.pi, 128):
            cos_phi = np.cos(phi)
            sin_phi = np.sin(phi)
            c_x = 2 + cos_theta
            c_y = sin_theta
            x = (
                c_x * (cos_B * cos_phi + sin_A * sin_B * sin_phi)- c_y * cos_A * sin_B
            ) * 2
            y = c_x * (sin_B * cos_phi - sin_A * cos_B * sin_phi) + c_y * cos_A * cos_B
            z = K2 + cos_A * c_x * sin_phi + c_y * sin_A
            z_1 = 1 / z
            p_x = int(canvas[0] / 2 + K1 * z_1 * x)
            p_y = int(canvas[1] / 2 - K1 * z_1 * y)
            L = (
                cos_phi * cos_theta * sin_B
                - cos_A * cos_theta * sin_phi
                - sin_A * sin_theta
                + cos_B * (cos_A * sin_theta - cos_theta * sin_A * sin_phi)
            )

            if L > 0 and z_1 > depth[p_x][p_y]:
                vertex[p_x][p_y] = L / np.sqrt(2)
                depth[p_x][p_y] = z_1

    return vertex


@np.vectorize
def get_fragment(value):
    if value == 0:
        return " "

    i = int(value * 11)

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
