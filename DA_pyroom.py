import random
import numpy as np
import pyroomacoustics as pra

from cfg_pyroom import mic_dict, src_dict, abs_coeff, fs, shoebox_vals, i_list


class DAwithPyroom(object):
    """
    Class for audio simulation using pyroom.
    input signal + 4 random crosstalk + background noise
    """
    def __init__(self, input_path, float_flag=True):
        """
        Start the class with the dataset path and turn the float32 flag for
        output format, False for int16 format
        """
        self.x_data = np.load(input_path)
        self.noise_data = np.load('./Noises_all.npy')
        self.float_flag = float_flag

        # Calculating largest size
        largest_idx = self.index_largest_entry()
        first_signal = self.x_data[largest_idx, :]
        if first_signal.dtype.kind != 'f':
            raise TypeError("'dtype' must be a floating point type")
        largest_signal = self.sim_single_signal(first_signal)
        self.max_length = len(largest_signal)

        # Numpy output array according to the desired output
        if self.float_flag:
            self.x_data_DA = np.zeros((self.x_data.shape[0], self.max_length),
                                      dtype='float32')
        else:
            self.x_data_DA = np.zeros((self.x_data.shape[0], self.max_length),
                                      dtype='int16')

    def index_largest_entry(self):
        """
        Iteratively extract the last values and check for the
        largest entry in the dataset.
        """
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
        Count using a non-optimized python alg the number of zeros
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

    def normalize_float32(self, audio_float32, gain_value=1):
        """
        Normalize float32 audio with the gain value provided
        """
        if audio_float32.dtype.kind != 'f':
            raise TypeError("'dtype' must be a floating point type")

        outmin = -1
        outmax = 1

        vmin = audio_float32.min()
        vmax = audio_float32.max()

        audio_float32 = (outmax - outmin)*(audio_float32 - vmin)/(vmax - vmin) \
            + outmin
        return audio_float32*gain_value

    def conv_2_int16(self, audio_float32, gain_value=1):
        """
        Converts float32 audio to int16 using
        the int16 min and max values by default
        """
        if audio_float32.dtype.kind != 'f':
            raise TypeError("'dtype' must be a floating point type")

        outmin = int(-32768*gain_value)
        outmax = int(32768*gain_value)

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
                                        dtype='float32')

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

        # Apply gain value and convert to required output format
        if self.float_flag:
            signal_offset = self.normalize_float32(others_current_audio,
                                                   gain_value=gain_value)
        else:
            signal_offset = self.conv_2_int16(others_current_audio,
                                              gain_value=gain_value)
        return signal_offset

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

        # Apply gain value and convert to required output format
        if self.float_flag:
            signal_offset = self.normalize_float32(noise_current_audio,
                                                   gain_value=gain_value)
        else:
            signal_offset = self.conv_2_int16(noise_current_audio,
                                              gain_value=gain_value)
        return signal_offset

    def sim_single_signal(self, input_signal, position=0, indx=0):
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

        others_audio1 = self.x_data[indx_others_1, :].astype('float32')
        others_audio2 = self.x_data[indx_others_2, :].astype('float32')
        others_audio3 = self.x_data[indx_others_3, :].astype('float32')
        noise_audio4 = self.noise_data[indx_noise_4, :].astype('float32')

        offset_value1 = self.gen_random_on_range(-0.4, 0.4)
        offset_value2 = self.gen_random_on_range(-0.4, 0.4)
        offset_value3 = self.gen_random_on_range(-0.4, 0.4)

        gain_value1 = self.gen_random_on_range(0.1, 0.2)
        gain_value2 = self.gen_random_on_range(0.001, 0.005)
        gain_value3 = self.gen_random_on_range(0.05, 0.1)
        gain_value4 = self.gen_random_on_range(0.1, 0.15)

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

        if self.float_flag:
            audio_original = self.normalize_float32(input_signal
                                                    [0:length_current_audio])
        else:
            audio_original = self.conv_2_int16(input_signal[0:length_current_audio])

        # Create 3D room and add sources
        room = pra.ShoeBox(shoebox_vals,
                           fs=fs,
                           absorption=abs_coeff,
                           max_order=16)
        room.add_source(src_dict["src_{}".format(i_list[position])],
                        signal=audio_original)
        room.add_source(src_dict["src_{}".format(i_list[position-3])],
                        signal=audio_offset1)
        room.add_source(src_dict["src_{}".format(i_list[position-2])],
                        signal=audio_offset2)
        room.add_source(src_dict["src_{}".format(i_list[position-1])],
                        signal=audio_offset3)
        room.add_source(src_dict["src_{}".format(i_list[position])],
                        signal=audio_offset4)

        # Define microphone array
        R = np.c_[mic_dict["mic_0"], mic_dict["mic_0"]]
        room.add_microphone_array(pra.MicrophoneArray(R, room.fs))

        # Compute image sources
        room.image_source_model()
        room.simulate()

        # Simulate audio
        raw_sim_audio = room.mic_array.signals[0, :]

        # Convert to required output format
        if self.float_flag:
            sim_audio = raw_sim_audio
        else:
            sim_audio = self.conv_2_int16(raw_sim_audio)

        print("{} index. {} {} {}, noise {}. Len {} sim {}".format(indx,
                                                                    indx_others_1,
                                                                    indx_others_2,
                                                                    indx_others_3,
                                                                    indx_noise_4,
                                                                    length_current_audio,
                                                                    len(sim_audio)))
        if position != 0:
            print("src_{} | src_{} src_{} src_{}".format(i_list[position],
                                                         i_list[position-3],
                                                         i_list[position-2],
                                                         i_list[position-1]))

        return sim_audio

    def sim_dataset(self, position=0):
        for indx in range(0, self.x_data.shape[0]):
            single_signal = self.x_data[indx, :]
            if single_signal.dtype.kind != 'f':
                raise TypeError("'dtype' must be a floating point type")
            single_x_DA = self.sim_single_signal(single_signal
                                                 .astype('float32'),
                                                 position,
                                                 indx)
            self.x_data_DA[indx, 0:len(single_x_DA)] = single_x_DA
        return self.x_data_DA
