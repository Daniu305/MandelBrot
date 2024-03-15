import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import matplotlib
matplotlib.use('macosx')


def MandelBrot(x, y, index=True):
    c = x + 1j * y
    z = np.zeros_like(c)
    n = np.zeros_like(c, dtype=np.int32)
    mask = np.ones_like(c, dtype=np.bool_) & index

    for i in range(500):
        z[mask] = z[mask] * z[mask] + c[mask]
        mask = np.abs(z) <= 2.5
        n[mask] += 1
        if not mask.any():
            break

    n[mask] = 1
    return n

class cMandelBrot:

    def __init__(self, x, y, Z):
        self.x = x
        self.hx = x[1] - x[0]
        self.y = y
        self.hy = y[0] - y[1]
        self.Z = Z
        self.precision = self.Z.shape[::-1]

    def newZ(self, xlim, ylim):
        xi0 = np.abs(self.x - xlim[0]).argmin()
        xi1 = np.abs(self.x - xlim[1]).argmin()
        yi0 = np.abs(self.y - ylim[1]).argmin()
        yi1 = np.abs(self.y - ylim[0]).argmin()

        self.Z = self.Z[yi0:yi1+1, xi0:xi1+1]
        self.x = self.x[xi0:xi1+1]
        self.y = self.y[yi0:yi1+1]
        return cMandelBrot(self.x, self.y, self.Z)

def plotMandelBrot(initial_xlim=(-2, 1), initial_ylim=(-1, 1)):
    MB = None
    def update_plot(xlim, ylim, precision=(int(65*1.5-0.5), 65), Zold=None, plot=True, target=1024):
        boolean = Zold is not None

        cprecision = ((1+boolean) * precision[0] - boolean, (1+boolean) * precision[1] - boolean)

        index = True
        if boolean: index = (np.arange(cprecision[0]) % 2 == 0) & (np.arange(cprecision[1]).reshape(-1, 1) % 2 == 0)

        xnew = np.linspace(xlim[0], xlim[1], cprecision[0])
        ynew = np.linspace(ylim[0], ylim[1], cprecision[1])
        if ynew[0] < ynew[-1]:
            ynew = ynew[::-1]

        Xnew, Ynew = np.meshgrid(xnew, ynew)
        Znew = MandelBrot(Xnew, Ynew, index=index)

        if boolean:
            cZ = np.zeros_like(Znew, dtype=Zold.dtype)
            cZ[::2, ::2] = Zold
            cZ += Znew
        else:
            cZ = Znew

        if plot:
            ax.clear()
            ax.set_title('Mandelbrot Set with precision = ' + str(cprecision))
            ax.set_xlabel('Real')
            ax.set_ylabel('Imaginary')
            ax.imshow(np.log(cZ), extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                      cmap='Spectral', aspect='equal')
            plt.draw()
            plt.pause(1)

        nonlocal MB
        MB = cMandelBrot(xnew, ynew, cZ)

        if abs(np.diff(xlim)) / MB.hx < target or abs(np.diff(ylim)) / MB.hy < target:
            update_plot(xlim, ylim, precision=MB.precision, Zold=MB.Z, plot=plot)

    fig, ax = plt.subplots(figsize=(10, 7))
    update_plot(initial_xlim, initial_ylim)

    refine_process = None

    def onselect(eclick, erelease):
        nonlocal refine_process
        nonlocal MB
        x0, y0 = eclick.xdata, eclick.ydata
        x1, y1 = erelease.xdata, erelease.ydata
        if refine_process is not None:
            refine_process.stop()
        refine_process = ax.clear()
        xlim = (min(x0, x1), max(x0, x1))
        ylim = (min(y0, y1), max(y0, y1))
        MB = MB.newZ(xlim, ylim)
        update_plot((MB.x[0], MB.x[-1]), (MB.y[0], MB.y[-1]), precision=MB.precision, Zold=MB.Z)

    def toggle_selector(event):
        if event.key in ['Q', 'q'] and toggle_selector.RS.active:
            toggle_selector.RS.set_active(False)
        if event.key in ['A', 'a'] and not toggle_selector.RS.active:
            toggle_selector.RS.set_active(True)

    toggle_selector.RS = RectangleSelector(ax, onselect, useblit=True, button=[1],
                                           minspanx=5, minspany=5, spancoords='pixels',
                                           interactive=True)
    plt.connect('key_press_event', toggle_selector)
    plt.show()

plotMandelBrot()

print('2')
print('weollllll6l')

