import os

# from pydub import AudioSegment
import sys
import time
from functools import partial

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft
from scipy.io import wavfile
from scipy.special import softmax
from tqdm import tqdm


def n_times_convolve(a, v, n, mode="same"):
    for i in range(n):
        a = np.convolve(a, v, mode=mode)
    return a


def linear_interpolation(x0, y0, x1, y1):
    return lambda x: (y0 - y1) / (x0 - x1) * (x - x0) + y0


def audibility_curve(domain):
    # the points below define an equal loudness curve for normal humans
    points = [
        [20, 99.85],
        [25, 93.94],
        [31.5, 88.17],
        [40, 82.63],
        [50, 77.78],
        [63, 73.08],
        [80, 68.48],
        [100, 64.37],
        [125, 60.59],
        [160, 56.70],
        [200, 53.41],
        [250, 50.40],
        [315, 47.58],
        [400, 44.98],
        [500, 43.05],
        [630, 41.34],
        [800, 40.06],
        [1000, 40.01],
        [1250, 41.82],
        [1600, 42.51],
        [2000, 39.23],
        [2500, 36.51],
        [3150, 35.61],
        [4000, 36.65],
        [5000, 40.01],
        [6300, 45.83],
        [8000, 51.80],
        [10000, 54.28],
        [12500, 51.49],
        [20000, 0],
    ]
    # edited equal loudness curve
    points = [
        [20, 100],
        [25, 99],
        [31.5, 98],
        [40, 97],
        [50, 96],
        [63, 95],
        [80, 94],
        [100, 93],
        [125, 60.59],
        [160, 56.70],
        [200, 53.41],
        [250, 50.40],
        [315, 47.58],
        [400, 44.98],
        [500, 43.05],
        [630, 41.34],
        [800, 40.06],
        [1000, 40.01],
        [1250, 41.82],
        [1600, 42.51],
        [2000, 39.23],
        [2500, 36.51],
        [3150, 35.61],
        [4000, 36.65],
        [5000, 40.01],
        [6300, 45.83],
        [8000, 51.80],
        [10000, 54.28],
        [12500, 51.49],
        [20000, 0],
    ]
    # points = [[0,180], [125,45], [250,27], [500,13.5], [750,9], [1000,7.5], [1500, 7.5], [2000,9], [3000, 11.5], [4000, 12], [6000, 16], [8000, 15.5], [30000,180]] # approximately normal human hearing range
    # points = [[0,180], [125,100], [250,75], [500,50], [750,50], [1000,50], [1500, 37.5], [2000,40], [3000, 30], [4000, 25], [6000, 20], [8000, 20], [30000,180]] # my hearing range
    for p in points:
        p[1] = 100 - p[1]
    curve = np.empty_like(domain)
    for i in range(len(domain)):
        for j in range(1, len(points)):
            x0, y0 = points[j - 1]
            x1, y1 = points[j]
            if domain[i] < x0:
                continue
            if domain[i] > x1:
                continue
            curve[i] = linear_interpolation(x0, y0, x1, y1)(domain[i])
    return curve / np.max(curve)


def main():

    # define all constants
    num_frames_per_fft = 8  # how many frames pass before the fft has completely moved on from a certain section of the wav file
    fft_sample_rate = 2**12  # how many samples to include in each fft
    top_htz = 8000  # highest frequency to display
    bottom_htz = 50  # lowest frequency to display

    top_max_color = (
        "w"  # these four are passed into matplotlib as the colors for the waveform
    )
    top_color = "w"  # they are in order from top to bottom

    bottom_color = "w"
    bottom_min_color = "w"

    alpha = 0.5

    epsilon = 0.01  # constant defining how small the convolution window is
    n_convolve = (
        1  # number of times to convolve the epsilon-ball with the fourier transform
    )

    decay_rate = 0.995  # decay rate of the min-max waves. smaller numbers decay faster.
    # numbers greater than 1 will grow exponentially
    transform_multiplier = (
        1.0  # each transform is multiplied by this to grow it a little
    )
    # this does mean that the highest amplitude will be off screen,
    # but it actually works when not in non-decay mode
    # i.e. when the decay rate is not 1
    max_amp_decay = 0.9999  # the running maximum amplitude decay rate
    max_amp_const = 500

    global_start_time = time.time()

    # os.system('clear')
    if len(sys.argv) < 2:
        raise ValueError("proper usage is: python3 file_transform.py [filename]")

    if len(sys.argv) > 2:
        if len(sys.argv) != 6:
            raise ValueError(
                "\n\nto specify waveform colors use format:\n\tpython3 file_transform.py [filename] [topmax] [topwave] [bottomwave] [bottommin]\nand all four colors must be specified"
            )
        top_max_color = sys.argv[2]
        top_color = sys.argv[3]
        bottom_color = sys.argv[4]
        bottom_min_color = sys.argv[5]
    print("colors:")
    print("\t" + top_max_color)
    print("\t" + top_color)
    print("\t" + bottom_color)
    print("\t" + bottom_min_color)

    filename = sys.argv[1]

    if filename[-3:] == "mp3":
        raise ValueError("mp3 conversion currently broken")
        # print("converting to wav file")
        # dest = filename[:-3] + "wav"
        # print(dest)
        # mp3 = AudioSegment.from_mp3(filename)
        # mp3.export(dest, format="wav")
    dest = filename[:-4]

    # get wavfile samples and rate, and normailze samples
    do_stereo = False
    rate, samples = wavfile.read(dest + ".wav")

    fps = 60
    ms_interval = 1000 / fps
    fft_sample_rate = int(ms_interval * rate * num_frames_per_fft / 1000)

    if len(samples.shape) != 2:
        if len(samples.shape) > 2:
            samples = np.sum(samples, axis=-1)
    else:
        samples_left = samples[:, 1]
        samples_right = samples[:, 0]
        samples = np.sum(samples, axis=-1)
        do_stereo = True
    print(rate)
    print(f"stereo: {do_stereo}")

    if not do_stereo:
        samples = samples / np.max(samples)
    else:
        samples_left = samples_left / np.max(samples_left)
        samples_right = samples_right / np.max(samples_right)
    # samples = samples[:len(samples)//5]

    num_samples = len(samples)
    duration = num_samples / rate
    print(f"duration of sound: {duration}")

    ms_interval = 1000 * fft_sample_rate / (rate * num_frames_per_fft)
    num_frames = (num_frames_per_fft * num_samples // fft_sample_rate) - 2
    print(f"num frames: {num_frames}")

    # preliminary fft to figure out the domain of the function

    samps = samples[:fft_sample_rate]
    samp_seconds = fft_sample_rate / rate
    freqs = fft(samps) / samp_seconds
    num_og_freqs = len(freqs)
    print(f"original number of frequencies: {num_og_freqs}")
    top = len(freqs) / samp_seconds
    print(f"top frequency: {top}")
    dom = np.linspace(0, top, len(freqs))[: len(freqs) // 2]

    mask = dom < top_htz
    dom = dom[mask]
    top_idx = len(dom)
    bottom_idx = 0

    for i in range(len(dom)):
        if dom[i] > bottom_htz:
            break
        else:
            bottom_idx = i

    num_freq = top_idx - bottom_idx
    print(f"number of frequencies after cleaning: {num_freq}")
    dom = dom[bottom_idx:]
    curve = audibility_curve(dom)

    ball = np.linspace(-0.999, 0.999, num_freq)
    v = (1 / epsilon) * np.exp(1 / ((ball / epsilon) ** 2 - 1))
    support = np.where(np.abs(ball) < epsilon, 1, 0)
    v = v * support
    v = v / np.max(v)
    # plt.plot(ball,v)
    # plt.show()
    plt.style.use("dark_background")
    # dpi = 250
    hd_size = (16, 9)
    fig = plt.figure(figsize=hd_size)
    ax = fig.add_subplot(111)
    ax.margins(x=0, y=0)
    ax.set_ylim((-1, 1))
    # ax.get_yaxis().set_visible(False)
    # ax.get_xaxis().set_visible(False)
    ax.axis("off")
    fig.tight_layout()

    dom = np.log2(
        dom
    )  # commenting this out will do plot things linearly (it just looks better if it's logarithmic on the frequency axis)

    (fourier,) = ax.plot(dom, np.zeros_like(dom), color=top_color)
    (nfourier,) = ax.plot(dom, np.zeros_like(dom), color=bottom_color)
    (max_fourier,) = ax.plot(dom, np.zeros_like(dom), color=top_max_color, alpha=alpha)
    (min_fourier,) = ax.plot(
        dom, np.zeros_like(dom), color=bottom_min_color, alpha=alpha
    )

    print(f"determining maximum amplitude (stereo={do_stereo}):")
    max_amp = 0
    max_amps_so_far = []
    if do_stereo:
        fourier_transforms_l = []
        fourier_transforms_r = []
    else:
        fourier_transforms = []

    fft_start_point = fft_sample_rate // num_frames_per_fft
    for i in tqdm(range(num_frames)):
        if do_stereo:
            samps_l = samples_left[
                i * fft_start_point : i * fft_start_point + fft_sample_rate
            ]
            freqs_l = fft(samps_l) / samp_seconds
            freqs_l = np.abs(freqs_l[bottom_idx:top_idx]) * curve
            freqs_l = np.nan_to_num(freqs_l, nan=0, posinf=0, neginf=0)
            freqs_l = n_times_convolve(freqs_l, v, n_convolve)

            samps_r = samples_right[
                i * fft_start_point : i * fft_start_point + fft_sample_rate
            ]
            freqs_r = fft(samps_r) / samp_seconds
            freqs_r = np.abs(freqs_r[bottom_idx:top_idx]) * curve
            freqs_r = np.nan_to_num(freqs_r, nan=0, posinf=0, neginf=0)
            freqs_r = n_times_convolve(freqs_r, v, n_convolve)
            local_max = max(np.max(freqs_r), np.max(freqs_l))
            max_amp *= max_amp_decay
            if local_max > max_amp:
                max_amp = local_max
            fourier_transforms_l.append(
                transform_multiplier * freqs_l / (max_amp + max_amp_const)
            )
            fourier_transforms_r.append(
                transform_multiplier * freqs_r / (max_amp + max_amp_const)
            )
            # print(f"max_amp={max_amp}",end="\r")
        else:
            samps = samples[i * fft_start_point : i * fft_start_point + fft_sample_rate]
            freqs = fft(samps) / samp_seconds
            freqs = np.abs(freqs[bottom_idx:top_idx]) * curve
            freqs = np.nan_to_num(freqs, nan=0, posinf=0, neginf=0)
            freqs = n_times_convolve(freqs, v, n_convolve)
            local_max = np.max(freqs)
            max_amp *= max_amp_decay
            if local_max > max_amp:
                max_amp = local_max
            fourier_transforms.append(transform_multiplier * freqs / max_amp)
    if do_stereo:
        max_freqs = list(np.zeros_like(freqs_r))
        min_freqs = list(np.zeros_like(freqs_l))
    else:
        max_freqs = list(np.zeros_like(freqs))
        min_freqs = list(np.zeros_like(freqs))
    print(f"animating {dest} at {1000/ms_interval:.1f} fps...")

    def update_stereo(
        frame, min_freqs, max_freqs
    ):  # ,max_amp_l,max_amp_r):#,num_og_freqs,bottom_idx,top_idx,curve,v,dom): # this currently breask if a non-stereo signal is passed in
        # samps_l = samples_left[frame*fft_start_point: frame*fft_start_point+fft_sample_rate]
        # freqs_l = np.real(fft(samps_l))/samp_seconds

        # freqs_l = np.abs(freqs_l[bottom_idx:top_idx])*curve
        # freqs_l = np.nan_to_num(freqs_l,nan=0,posinf=0,neginf=0)
        # freqs_l = n_times_convolve(freqs_l,v,n_convolve)
        freqs_l = fourier_transforms_l[frame]
        # freqs_l = transform_multiplier*freqs_l/max_amps_so_far[frame]

        # samps_r = samples_right[frame*fft_start_point: frame*fft_start_point+fft_sample_rate]
        # freqs_r = np.real(fft(samps_r))/samp_seconds

        # freqs_r = np.abs(freqs_r[bottom_idx:top_idx])*curve
        # freqs_r = np.nan_to_num(freqs_r,nan=0,posinf=0,neginf=0)
        # freqs_r = n_times_convolve(freqs_r,v,n_convolve)
        freqs_r = fourier_transforms_r[frame]
        # freqs_r = transform_multiplier*freqs_r/max_amps_so_far[frame]

        for i in range(len(max_freqs)):
            max_freqs[i] = max_freqs[i] * decay_rate
            if max_freqs[i] < freqs_r[i]:
                max_freqs[i] = freqs_r[i]

            min_freqs[i] = min_freqs[i] * decay_rate
            if min_freqs[i] > -freqs_l[i]:
                min_freqs[i] = -freqs_l[i]

        fourier.set_ydata(freqs_r)
        nfourier.set_ydata(-freqs_l)
        max_fourier.set_ydata(max_freqs)
        min_fourier.set_ydata(min_freqs)

        return fourier, nfourier, max_fourier, min_fourier

    def update_mono(frame):
        # samps = samples[frame*fft_start_point: frame*fft_start_point+fft_sample_rate]
        # freqs = np.real(fft(samps))/samp_seconds

        # freqs = np.abs(freqs[bottom_idx:top_idx])*curve
        # freqs = np.nan_to_num(freqs,nan=0,posinf=0,neginf=0)
        # freqs = n_times_convolve(freqs,v,n_convolve)
        # freqs = transform_multiplier*freqs/max_amp
        freqs = fourier_transforms[frame]

        for i in range(len(max_freqs)):
            max_freqs[i] = max_freqs[i] * decay_rate
            if max_freqs[i] < freqs[i]:
                max_freqs[i] = freqs[i]

            min_freqs[i] = min_freqs[i] * decay_rate
            if min_freqs[i] > -freqs[i]:
                min_freqs[i] = -freqs[i]

        fourier.set_ydata(freqs)
        nfourier.set_ydata(-freqs)
        max_fourier.set_ydata(max_freqs)
        min_fourier.set_ydata(min_freqs)
        return fourier, nfourier, max_fourier, min_fourier

    animation.writer = animation.writers["ffmpeg"]
    plt.ioff()
    if do_stereo:
        ani = animation.FuncAnimation(
            fig,
            partial(update_stereo, min_freqs=min_freqs, max_freqs=max_freqs),
            frames=num_frames,
            interval=ms_interval,
            blit=True,
        )
    else:
        ani = animation.FuncAnimation(
            fig,
            partial(update_mono),
            frames=num_frames,
            interval=ms_interval,
            blit=True,
        )
    start_time = time.time()
    pbar = tqdm(total=num_frames)

    def progress(i, n):
        pbar.update()

    ani.save(
        dest + ".mp4", progress_callback=progress
    )  # lambda i,n: print(f'  {100*((i+1)/n):.3f}% eta: {(time.time()-start_time)*((n-i)/(i+1)):.0f} sec   ',end='\r'))
    pbar.close()
    plt.close()
    os.system(
        f"ffmpeg -y -i {dest}.mp4 -i {dest}.wav -c:v copy -c:a aac {dest}_animated.mp4"
    )
    os.system(f"rm {dest}.mp4")
    os.system("clear")
    print("done")
    print(
        f"\ncomputation time::sound time: {(time.time() - global_start_time)/duration}"
    )


if __name__ == "__main__":
    main()
