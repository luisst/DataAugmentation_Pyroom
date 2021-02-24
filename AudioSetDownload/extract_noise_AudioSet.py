# -*- coding: utf-8 -*-
import csv


lbl_speechNoise = '/m/07qfr4h'

new_csv = []
new_csv.append(["YTID", "start_seconds", "end_seconds"])

with open('unbalanced_train_segments.csv', 'rt') as f:
    csv_reader = csv.reader(f,
                            quotechar='"',
                            delimiter=',',
                            quoting=csv.QUOTE_ALL,
                            skipinitialspace=True)

    for line in csv_reader:
        if lbl_speechNoise in line[3]:
            print('found!')
            line.pop()
            print(line)
            new_csv.append(line)

with open('noiseCSV.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(new_csv)
