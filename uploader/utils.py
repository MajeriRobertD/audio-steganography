import matplotlib.pyplot as plt
import wave
import numpy as np
import base64
from io import BytesIO

def get_graph():
    the_buffer = BytesIO()
    plt.savefig(the_buffer, format = 'png' )
    the_buffer.seek(0)
    image_png = the_buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    the_buffer.close()
    return graph

def get_plot(x,y):
    plt.switch_backend('AGG')
    plt.figure(figsize = (10,5))
    plt.title("sound wave")
    plt.plot(x,y)
    plt.xlabel("Time(seconds)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    graph = get_graph()
    return graph

def get_coords(filename):
    sound = wave.open(filename,"r")
    sound_wave = sound.readframes(-1)
    signal_sound = np.frombuffer(sound_wave, np.int16)
    framerate_sound = sound.getframerate()

    signal_sound.shape = -1,2
    signal_sound = signal_sound.T

    n_frames = sound.getnframes()
    duration = 1/float(framerate_sound)

    t_sq = np.arange(0,n_frames/float(framerate_sound), duration)

    return [t_sq, signal_sound[0]]
