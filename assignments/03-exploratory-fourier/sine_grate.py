import numpy as np
import matplotlib.pyplot as plt


def create_sine_grate(freq: float, cycles: int) -> np.ndarray:
    """Creates a 2D mesh grid with a sine wave pattern."""
    size = 20 * cycles
    x = np.linspace(0, 2 * np.pi * cycles, size)
    xx, _ = np.meshgrid(x, x)
    mesh = 100 * np.sin(freq * xx)
    return mesh


def plot_fft_sine_grate(freq: float, cycles: int) -> None:
    """Plots a 2D mesh grid with a sine wave pattern."""
    mesh = create_sine_grate(freq, cycles)
    fft_result = np.fft.fftshift(np.fft.fft2(mesh))
    magnitude_spectrum = np.abs(fft_result)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].imshow(mesh, cmap="gray")
    ax[0].set_title(f"Sine Grating: {freq} Hz, {cycles} Cycles")

    ax[1].imshow(magnitude_spectrum, cmap="gray")
    ax[1].set_title("FFT Magnitude Spectrum")
    fig.savefig(f"results/sine_grate_fft_{freq}Hz.png")
    plt.show()


if __name__ == "__main__":
    plot_fft_sine_grate(1, 5)
    plot_fft_sine_grate(2, 5)
    plot_fft_sine_grate(3, 5)
