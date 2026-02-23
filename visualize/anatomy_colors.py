import os
import numpy as np


class AnatomyColors:
    def __init__(self):
        self.tooth_colors = np.loadtxt(os.path.join(os.path.dirname(__file__), 'tooth_rgb.txt'), dtype=int)
        self.anatomy_colors = np.loadtxt(os.path.join(os.path.dirname(__file__), 'anatomy_rgb.txt'), dtype=int)
        self.district_colors = None

    def get_color(self, i, norm=False):
        """
        Get generic anatomy color by index in RGB format

        See https://www.slicer.org/wiki/Slicer3:2010_GenericAnatomyColors for a full table

        :param i: Index, 0 is background (#000000)
        :type i: int
        :param norm: The color form. If true, the range is 0 ~ 255, else 0.0 ~ 1.0. <br> Default is <code>False</code>.
        :type norm: bool
        :return: A 3-dimensional NumPy array in order of [R,G,B].
        :rtype: ndarray
        """
        if norm:
            return self.anatomy_colors[i % len(self.anatomy_colors)] / 255.0
        else:
            return self.anatomy_colors[i % len(self.anatomy_colors)]

    def get_arr_tooth_color(self, arr, norm=False):
        color = np.zeros((len(arr), 3), dtype=np.uint8 if not norm else np.float32)
        for i in np.unique(arr):
            color[arr == i] = self.get_tooth_district_color(i, norm)
        return color

    def get_tooth_color(self, i, norm=False):
        if i == 0:
            return np.array([234, 234, 234]) / 255 if norm else np.array([234, 234, 234])
        if norm:
            return self.tooth_colors[(i - 1) % 8 + 1] / 255.0
        else:
            return self.tooth_colors[(i - 1) % 8 + 1]

    def get_tooth_continuous_color_lut(self):
        lut = np.zeros((33, 3))
        for i in range(33):
            if i == 0:
                lut[i] = self.get_tooth_district_color(i)
            else:
                lut[i] = self.get_tooth_district_color(((i - 1) // 8 + 1) * 10 + (i - 1) % 8 + 1)
        return lut

    def get_tooth_district_color(self, i, norm=False):
        """
        Get tooth color by quadrant.

        1X: blue; 2X: red; 3X: green; 4X: pink

        :param i: Tooth number in FDI notation
        :type i: int
        :param norm: Whether to normalize to [0, 1]
        :type norm: bool
        :return:
        :rtype:
        """
        if i == 0:
            return np.array([234, 234, 234]) / 255 if norm else np.array([234, 234, 234])
        base_colors = np.array([
            [57, 74, 244],
            [255, 75, 75],
            [55, 158, 50],
            [223, 91, 236]
        ])
        base_colors = base_colors / 255 if norm else base_colors
        district = (i // 10 - 1) % 4
        return np.asarray(base_colors[district] * 0.2 + 0.8 * self.get_tooth_color(i % 10, norm), dtype=np.int32 if not norm else np.float32)

    def get_district_label_by_color(self, color, norm=True, tolerance=0.01):
        if not norm:
            color = color / 255
        if self.district_colors is None:
            self.district_colors = []
            self.district_colors.append(self.get_tooth_district_color(0, norm=True))
            for i in range(1, 5):
                for j in range(1, 9):
                    self.district_colors.append(self.get_tooth_district_color(i * 10 + j, norm=True))
            self.district_colors = np.array(self.district_colors)
        color = np.expand_dims(color, 0).repeat(33, 0)
        diff = np.sum(np.abs(color - self.district_colors), axis=-1)
        if np.sum(diff < tolerance) > 0:
            idx = np.argwhere(diff < tolerance).squeeze()
            if idx == 0:
                return 0
            return ((idx - 1) // 8 + 1) * 10 + (idx - 1) % 8 + 1
        else:
            print(diff)
            return 0


if __name__ == '__main__':
    colors = AnatomyColors()
    for i in range(1, 5):
        print('<script>function rgbToHex(r, g, b) { return "#" + ( ( 1 << 24) + (r << 16) + (g << 8) + b).toString ( 16 ).slice ( 1 ); }</script>')
        print('<div style="display: flex">')
        for j in range(1, 9):
            c = colors.get_tooth_district_color(i * 10 + j)
            print(f'<div style="width: 160px; padding: 16px; background: rgb({c[0]}, {c[1]}, {c[2]})"><div>{i}{j}</div>'
                  f'<div>RGB {c[0]} {c[1]} {c[2]}</div>'
                  f'<div id="c{i}{j}"></div>'
                  f'<script>document.getElementById("c{i}{j}").innerHTML = rgbToHex({c[0]}, {c[1]}, {c[2]})</script>'
                  f'</div>')
        print('</div>')
