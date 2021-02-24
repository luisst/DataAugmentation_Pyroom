

mic_dict = {"mic_0": [6, 10, 0],
            "mic_1": [6.1, 10, 0]}

src_dict = {"src_0": [7, 10.5, 0.5],
            "src_1": [6.5, 9, 0.5],
            "src_2": [6.5, 10.5, 0.8],
            "src_3": [7.5, 8.5, 0.3]}

shoebox_vals = [12, 14, 2.5]

abs_coeff = 0.7

fs = 16000

v = list(src_dict.values())
room_x_max = 0
room_y_max = 0
room_z_max = 0
for vals in v:
    if room_x_max < vals[0]:
        room_x_max = vals[0]
    if room_y_max < vals[1]:
        room_y_max = vals[1]
    if room_z_max < vals[2]:
        room_z_max = vals[2]

src_max = [room_x_max, room_y_max, room_z_max]
