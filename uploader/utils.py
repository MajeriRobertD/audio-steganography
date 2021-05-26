import matplotlib.pyplot as plt
import wave
import numpy as np
import base64
from io import BytesIO
from Cryptodome.Cipher import AES
from Cryptodome.Util.Padding import pad, unpad
BLOCK_SIZE = 32 # Bytes

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
    
