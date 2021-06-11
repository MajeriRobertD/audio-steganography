import matplotlib.pyplot as plt
import wave
import numpy as np
from math import atan2, floor
import base64
from io import BytesIO
from Cryptodome.Cipher import AES
from Cryptodome.Util.Padding import pad, unpad
BLOCK_SIZE = 32 # Bytes
# to capture console args
import sys, getopt
# math functions
from math import *
import cmath

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

def aes_encrypt(message, key):
    cipher = AES.new(key.encode('utf8'), AES.MODE_ECB)
    msg = cipher.encrypt(pad(message.encode('UTF-8'), BLOCK_SIZE))
    return msg

def aes_decrypt(cipher, key):
    decipher = AES.new(key.encode('utf8'), AES.MODE_ECB)
    msg_dec = decipher.decrypt(cipher)
    return unpad(msg_dec, BLOCK_SIZE).decode("UTF-8")


def encode(filename, message, key):
    # We will use wave package available in native Python installation to read and write .wav audio file

    # read wave audio file
    song = wave.open(filename, mode='rb')
    # Read frames and convert to byte array
    frame_bytes = bytearray(list(song.readframes(song.getnframes())))
    
    # The "secret" text message
    msg =message
    msg = aes_encrypt(msg, key)

    string = str(msg, 'latin-1')
    # Append dummy data to fill out rest of the bytes. Receiver shall detect and remove these characters.
    string = string + int((len(frame_bytes)-(len(string)*8*8))/8) *'#'
    # Convert text to bit array
    bits = list(map(int, ''.join([bin(ord(i)).lstrip('0b').rjust(8,'0') for i in string])))
    
    # Replace LSB of each byte of the audio data by one bit from the text bit array
    for i, bit in enumerate(bits):
        frame_bytes[i] = (frame_bytes[i] & 254) | bit
    # Get the modified bytes
    frame_modified = bytes(frame_bytes)
    
    # Write bytes to a new wave audio file
    with wave.open('/home/robert/licenta/app/licenta/media/song_embedded.wav', 'wb') as fd:
        fd.setparams(song.getparams())
        fd.writeframes(frame_modified)
    song.close()


def decode(filename, key):    
    # Use wave package (native to Python) for reading the received audio file
    import wave
    song = wave.open(filename, mode='rb')
    # Convert audio to byte array
    frame_bytes = bytearray(list(song.readframes(song.getnframes())))

    # Extract the LSB of each byte
    extracted = [frame_bytes[i] & 1 for i in range(len(frame_bytes))]
    # Convert byte array back to string
    string = "".join(chr(int("".join(map(str,extracted[i:i+8])),2)) for i in range(0,len(extracted),8))
    # Cut off at the filler characters
    decoded = string.split("###")[0]
    decoded = decoded.encode('latin-1')
    msg_dec = aes_decrypt(decoded,key)
    return msg_dec

    # Print the extracted text
    # return "Sucessfully decoded: "+decoded
    song.close()




    ###############################################PHASE ENCODING##############

## HELPER METHODS   ##
def chunks(l, n):
    """
    Split list into chunks.
    source: http://stackoverflow.com/a/1751478

    :param l: list to split
    :param n: chunk size
    :return: [[],[],[],...]
    """
    n = max(1, n)
    return [l[i:i + n] for i in range(0, len(l), n)]


def arg(z):
    """
    Argument of a complex number
    :param z: complex number
    :return:  arg(z)
    """
    return atan2(z.imag, z.real)


def vec_2_str(vec):
    """
    Convert vector of integers to string.
    :param vec: [int, int, ...]
    :return: string
    """
    char_vec = [chr(i) for i in vec]
    return ''.join(char_vec)


def str_2_vec(str):
    """
    Convert vector of integers to string.
    :param str: string
    :return:    [int, int, int, ...]
    """
    return [ord(i) for i in str]


def d_2_b(x, size=8):
    """
    Convert decimal to byte list
    :param x:    decimal
    :param size: the size of byte list
    :return: e.g. [0, 0, 1, ...]
    """
    s = np.sign(x)
    v = size * [None]
    for i in range(0, size):
        v[i] = abs(x) % 2
        x = int(floor(abs(x)/2.0))
    return s * v


def b_2_d(x):
    """
    Convert byte list to decimal
    :param x:   byte list
    :return:    decimal
    """
    s = 0
    for i in range(0, len(x)):
        s += x[i]*2**i
    return s
    

### helper methods for reading wav files:

def wav_load(file_name):
    """
    Load wav file
    source: http://stackoverflow.com/a/2602334
    :param file_name: file name
    :return: (number of channels,
             bytes per sample,
             sample rate,
             number of samples,
             compression type,
             compression name
             ),
             (left, right)

             Note: if signal is mono than left = right
    """
    wav = wave.open(file_name, "r")
    (nchannels, sampwidth, framerate, nframes, comptype, compname) = wav.getparams()
    frames = wav.readframes(nframes * nchannels)
    #out = struct.unpack_from("%dh" % nframes * nchannels, frames)
    left, right = audio_decode(frames, nchannels)
    wav.close()

   
    return (nchannels, sampwidth, framerate, nframes, comptype, compname), (left, right)


def wav_save(file_name, samples, nchannels=2, sampwidth=2, framerate=44100, nframes=None, comptype='NONE', compname='not compressed'):
    """
    Save wav file.
    :param file_name: file name
    :param samples:   samples = (left, right)
    :param nchannels: number of channels
    :param sampwidth: bytes per sample
    :param framerate: sample rate
    :param nframes:   number of frames
    :param comptype:  compression type
    :param compname:  compression name
    """
    wv = wave.open(file_name, 'w')
    wv.setparams((nchannels, sampwidth, framerate, nframes, comptype, compname))
    # if nchannels == 2:
    #     data = [None]*(len(samples[0])+len(samples[1]))
    #     data[::2] = samples[0]
    #     data[1::2] = samples[1]
    # else:
    #     data = samples[0]
    #frames = struct.pack("%dh" % len(data), *data)
    frames = audio_encode(samples)
    wv.writeframesraw(frames)
    wv.close()


def audio_decode(in_data, channels):
    result = np.fromstring(in_data, dtype=np.int16)
    chunk_length = len(result) // channels
    output = np.reshape(result, (chunk_length, channels))
    # output = result.values.reshape([chunk_length, channels])
    l, r = np.copy(output[:, 0]), np.copy(output[:, 1])

    return l.tolist(), r.tolist()


def audio_encode(samples):
    l, r, = samples
    interleaved = np.array([l, r]).flatten('F')
    out_data = interleaved.astype(np.int16).tostring()
    return out_data


###### phase encoding methods:




def hide(source, destination, message):
    """
    :param source:  source stego container filename
    :param destination:    dest stego container filename
    :param message:       message to hide
    :return:        segment_width - segment width
    """
    # read wav file
    print ('reading wave container...')
    (nchannels, sampwidth, framerate, nframes, comptype, compname),\
    (left, right) = wav_load(source)
    # select channel to hide message in
    container = left
    container_len = len(container)
    # --------------------------------------
    # prepare container
    # --------------------------------------
    print ('preparing container...')
    message_len = 8 * len(message)          # msg len in bits
    v = int(ceil(log(message_len, 2)+1))    # get v from equation: 2^v >= 2 * message_len
    segment_width = 2**(v+1)                # + 1 to reduce container distortion after msg integration
    segment_count = int(ceil(container_len / segment_width))    # number of segments to split container in
    # add silence if needed
    if segment_count > container_len / segment_width:
        container = [(container[i] if i < container_len else 0) for i in range(0, segment_count*segment_width)]
    container_len = len(container)          # new container length
    # split signal in 'segment_count' segments with 'segment_width' width
    segments = chunks(container, segment_width)
    # --------------------------------------
    # apply FFT
    # --------------------------------------
    print ('performing fft transform...')
    delta = [np.fft.rfft(segments[n]) for n in range(0, segment_count)]  # -> segment_width / 2 + 1
    # extract amplitudes
    vabs = np.vectorize(abs)    # apply vectorization
    amps = [vabs(delta[n]) for n in range(0, segment_count)]
    # extract phases
    varg = np.vectorize(arg)    # apply vectorization
    phases = [varg(delta[n]) for n in range(0, segment_count)]
    # --------------------------------------
    # save phase subtraction
    delta_phases = segment_count*[None]
    delta_phases[0] = 0 * phases[0]
    def sub (a, b): return a - b
    vsub = np.vectorize(sub)
    for n in range(1, segment_count):
        delta_phases[n] = vsub(phases[n], phases[n-1])
    # --------------------------------------
    # integrate msg, modify phase
    print( 'msg integration...')
    msg_vec = str_2_vec(message)
    msg_bits = [d_2_b(msg_vec[t]) for t in range(0, len(message))]
    msg_bits = [item for sub_list in msg_bits for item in sub_list]  # msg is a list of bits now

    segment_width_half = segment_width // 2

    phase_data = (segment_width_half + 1) * [None]  # preallocate list where msg will be stored
    for k in range(0, segment_width_half + 1):
        if k <= len(msg_bits):
            if k == 0 or k == segment_width_half:   # do not modify phases at the ends
                phase_data[k] = phases[0][k]
            if 0 < k < segment_width_half:          # perform integration begining with the hi-freq. components
                if msg_bits[k-1] == 1:
                    phase_data[segment_width_half+1-k] = -pi / 2.0
                elif msg_bits[k-1] == 0:
                    phase_data[segment_width_half+1-k] = pi / 2.0
        if k > len(msg_bits):                       # original phase
            phase_data[segment_width_half+1-k] = phases[0][segment_width_half+1-k]
    phases_modified = [phase_data]
    for n in range(1, segment_count):
        phases_modified.append((phases_modified[n-1] + delta_phases[n]))
    # --------------------------------------
    # convert data back to the frequency domain: amplitude * exp(1j * phase)
    def to_frequency_domain (amp, ph): return amp * cmath.exp(1j * ph)
    vto_fft_result = np.vectorize(to_frequency_domain)
    delta_modified = [vto_fft_result(amps[n], phases_modified[n]) for n in range(0, segment_count)]
    # restore segments
    segments_modified = [np.fft.irfft(delta_modified[n]) for n in range(0, segment_count)]
    # join segments
    container_modified = [item for sub_list in segments_modified for item in sub_list]
    # sync the size of unmodified channel with the size of modified one
    right_synced = len(container_modified) * [None]
    for i in range(0, len(container_modified)):
        if i < len(right):
            right_synced[i] = right[i]
        else:
            right_synced[i] = 0
    # --------------------------------------
    # save stego container with integrated message in freq. scope as wav file
    print ('saving stego container...')
    wav_save(destination, (container_modified, right_synced),
                    nchannels, sampwidth, framerate, nframes, comptype, compname)
    # to recover the message the one must know the segment width, used in the process
    print ("\nDone.\n")
    print(segment_width)
    return segment_width


def recover(source, segment_width):
    """
    :param source: filename for the file with integrated message
    :param segment_width: segment width
    :return: message
    """
    # read wav file with integrated message
    print ('reading wave container...')
    (nchannels, sampwidth, framerate, nframes, comptype, compname),\
    (left, right) = wav_load(source)
    container = left    # take left channel for msg recovering
    container_len = len(container)
    print("container len:", container_len)
    # --------------------------------------
    # prepare container
    print ('preparing container...')
    segment_count = int(container_len / segment_width)
    # split signal in 'segment_count' segments with 'width' width
    segments = chunks(container, segment_width)
    # --------------------------------------
    # apply FFT
    print ('performing fft transform...')
    delta = [np.fft.rfft(segments[0])]
    # extract phases
    varg = np.vectorize(arg)    # apply vectorization
    phases = [varg(delta[0])]
    phases_0_len = len(phases[0])
    # --------------------------------------
    # recover message
    print ('recovering message...')
    b = []
    for t in range(0, segment_width//2):
        d = phases[0][phases_0_len-1-t]
        print(d)
        if d <= -pi / 3.0:
            b.append(1)
        elif d >= pi / 3.0:
            b.append(0)
        else:
            break
    msg_bits_len = int(floor(len(b) / 8.0))
    
    msg_bits_splitted = chunks(b, 8)
    msg_vec = []
    for i in range(0, msg_bits_len):
        msg_vec.append(b_2_d(msg_bits_splitted[i]))
    message = vec_2_str(msg_vec)
    print ("\nDone.\n")
    return message

