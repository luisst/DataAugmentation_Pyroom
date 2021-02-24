import random
import numpy as np
import pyroomacoustics as pra

from cfg_pyroom import mic_dict, src_dict, abs_coeff, fs, shoebox_vals

# is this slower because I am copying the large Audios_numpy into X_data?

# activate autocompletion in python

# use terminator or shell pop up?

# https://github.com/syl20bnr/spacemacs/issues/10638


class DAwithPyroom(object):
    """
    Class for audio simulation using pyroom.
    input signal + 4 random crosstalk + background noise
    """
    def __init__(self, input_path):
        self.x_data = np.load(input_path)
        self.noise_data = np.load('./Noises_all.npy')

        # Calculating largest size
        largest_idx = self.index_largest_entry()
        first_signal = self.x_data[largest_idx, :]
        largest_signal = self.sim_single_signal(first_signal)
        self.max_length = len(largest_signal)
        self.x_data_DA = np.zeros((self.x_data.shape[0], self.max_length),
                                  dtype='int16')
        
    
    def index_largest_entry(self):
        idx = 1
        while True:
            last_column = self.x_data[:,-idx]
            if np.count_nonzero(last_column) != 0:
                array_nonzero = np.nonzero(last_column)
                print("Largest at entry {}".format(array_nonzero[0][0]))
                return array_nonzero[0][0]
            idx = idx + 1
                

    def get_padding_value(self, signal):
        """
        count using a non-optimized python alg the number of zeros
        at the end of the numpy array
        """
        # real_length is initialized with the total length of the signal
        real_length = int(len(signal))
        while signal[real_length - 1] == 0:
            real_length = real_length - 1
        return real_length - 1

    def gen_random_on_range(self, lower_value, max_value):
        """
        generates a random value between lower_value and max_value.
        """
        return round(lower_value + random.random()*(max_value - lower_value),
                     2)

    def conv_2_int16(self, audio_float32, id_audio='empty', outmin=-32768, outmax=32767):
        """
        converts float32 audio to int16 using
        the int16 min and max values by default
        """
        # outmin = -32768
        # outmax = 32767

        vmin = audio_float32.min()
        vmax = audio_float32.max()

        audio_int16 = (outmax - outmin)*(audio_float32 - vmin)/(vmax - vmin) \
            + outmin
        audio_int16 = audio_int16.astype('int16')
        return audio_int16

    def audio_mod(self, signal, gain_value, offset_value,
                  length_current_audio):
        """
        Modifies the signal with a gain_value (percentage) and
        offset_value(percentage) and converts output to int16
        """

        signal_length = signal.shape[0]
        signal_offset = np.zeros_like(signal)
        others_current_audio = np.zeros((length_current_audio),
                                        dtype='float64')

        # Calculate the offset factors at the start and end
        factor1 = int(signal_length*((abs(offset_value) - offset_value)/2))
        factor2 = int(signal_length*((abs(offset_value) + offset_value)/2))

        # Apply the offset factors
        signal_offset[factor1: (signal_length - factor2)] = \
            signal[factor2: (signal_length - factor1)]

        # Trim offset signal to the real length of the audio
        if signal_length > length_current_audio:
            others_current_audio = signal_offset[0:length_current_audio]
        else:
            others_current_audio[0:signal_length] = signal_offset

        # Calculate gain value (from 0 to 1)
        outmin = int(-32768*gain_value)
        outmax = int(32767*gain_value)

        # Apply gain value and convert to int16
        signal_offset_int16 = self.conv_2_int16(others_current_audio,
                                                outmin=outmin,
                                                outmax=outmax)
        return signal_offset_int16

    def noise_mod(self, noise, gain_value, length_current_audio):
        """
        Modifies the signal with a gain_value (percentage) and
        offset_value(percentage) and converts output to int16
        """

        # Calculate the offset factors at the start
        noise_length = noise.shape[0]
        noise_current_audio = np.zeros((length_current_audio), dtype='float64')

        # Accomodate noise audios within the signal audio length
        if noise_length > length_current_audio:
            noise_current_audio = noise[0:length_current_audio]
        else:
            noise_current_audio[0:noise_length] = noise

        # Calculate gain value (from 0 to 1)
        outmin = int(-32768*gain_value)
        outmax = int(32767*gain_value)

        # Apply gain value and convert to int16
        signal_offset_int16 = self.conv_2_int16(noise_current_audio,
                                                outmin=outmin,
                                                outmax=outmax)
        return signal_offset_int16


    def sim_single_signal(self, input_signal, indx = 0):
        """
        Pyroomacoustics simulation with 3 random other audios
        from the same x_data + 1 noise from AudioSet
        """

        length_current_audio = self.get_padding_value(input_signal)

        # Load others audios
        indx_others_1 = random.randint(0, len(self.x_data)-1)
        indx_others_2 = random.randint(0, len(self.x_data)-1)
        indx_others_3 = random.randint(0, len(self.x_data)-1)
        indx_noise_4 = random.randint(0, len(self.noise_data)-1)

        

        others_audio1 = self.x_data[indx_others_1, :]
        others_audio2 = self.x_data[indx_others_2, :]
        others_audio3 = self.x_data[indx_others_3, :]
        noise_audio4 = self.noise_data[indx_noise_4, :]

        offset_value1 = self.gen_random_on_range(-0.4, 0.4)
        offset_value2 = self.gen_random_on_range(-0.4, 0.4)
        offset_value3 = self.gen_random_on_range(-0.4, 0.4)

        gain_value1 = self.gen_random_on_range(0.1, 0.2)
        gain_value2 = self.gen_random_on_range(0.001, 0.005)
        gain_value3 = self.gen_random_on_range(0.05, 0.1)
        gain_value4 = self.gen_random_on_range(0.1, 0.17)

        audio_offset1 = self.audio_mod(others_audio1,
                                       gain_value1, offset_value1,
                                       length_current_audio)
        audio_offset2 = self.audio_mod(others_audio2,
                                       gain_value2, offset_value2,
                                       length_current_audio)
        audio_offset3 = self.audio_mod(others_audio3,
                                       gain_value3, offset_value3,
                                       length_current_audio)
        audio_offset4 = self.noise_mod(noise_audio4,
                                       gain_value4,
                                       length_current_audio)

        audio_original = self.conv_2_int16(input_signal[0:length_current_audio])

        # Create 3D room and add sources
        room = pra.ShoeBox(shoebox_vals,
                           fs=fs,
                           absorption=abs_coeff,
                           max_order=16)
        room.add_source(src_dict["src_0"], signal=audio_original)
        room.add_source(src_dict["src_1"], signal=audio_offset1)
        room.add_source(src_dict["src_2"], signal=audio_offset2)
        room.add_source(src_dict["src_3"], signal=audio_offset3)
        room.add_source(src_dict["src_0"], signal=audio_offset4)

        # Define microphone array
        R = np.c_[mic_dict["mic_0"], mic_dict["mic_0"]]
        room.add_microphone_array(pra.MicrophoneArray(R, room.fs))

        # Compute image sources
        room.image_source_model()
        room.simulate()

        # Simulate audio and convert to int16
        full_sim_audio_int16 = self.conv_2_int16(room.mic_array.signals[0, :])
        print("{} index. {} {} {}, noise {}. Len {} sim {}".format(indx, 
                                                    indx_others_1,
                                                 indx_others_2,
                                                 indx_others_3,
                                                 indx_noise_4,
                                                 length_current_audio,
                                                 len(full_sim_audio_int16)))

        return full_sim_audio_int16

    def sim_dataset(self):
        for indx in range(0,self.x_data.shape[0]):
            single_x_DA = self.sim_single_signal(self.x_data[indx, :], indx)
            self.x_data_DA[indx, 0:len(single_x_DA)] = single_x_DA
        return self.x_data_DA
