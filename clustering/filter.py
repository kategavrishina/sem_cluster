import pandas as pd
import sys
import os
import shutil

clusters_dir = sys.argv[1]
filter_type = sys.argv[2]

if not os.path.exists(clusters_dir):
    print('Введите название существующей директории')

filter_type = int(filter_type)


if filter_type == 1:
    log = open('filter1_log.txt', 'w+')
    print('Фильтрация 1: удаление малочиселнных кластеров\n', file=log)
elif filter_type == 2:
    log = open('filter2_log.txt', 'w+')
    print('Фильтрация 2: добавление малочисленных кластеров к самому многочисленному\n', file=log)
else:
    print('Выберите один из предложенных типов фильтрации: 1 - удаление малочиселнных кластеров,'
          '2 - добавление малочисленных кластеров к самому многочисленному')

out_dir = clusters_dir + '_filter' + str(filter_type)

if os.path.exists(out_dir):
    shutil.rmtree(out_dir)

os.makedirs(out_dir)

for file in os.listdir(clusters_dir):
    print(file, file=log)
    data = pd.read_csv(os.path.join(clusters_dir, file), sep='\t')
    count_one = 0
    counts = data['cluster'].value_counts().to_dict()
    for k, v in counts.items():
        if v == 1:
            count_one += 1
            if filter_type == 1:
                data = data[data.cluster != k]
            elif filter_type == 2:
                data['cluster'] = data['cluster'].replace(k, 0)
    print(f'{count_one} кластеров было {"упразднено" if filter_type == 1 else "добавлено в кластер 0"}', file=log)
    data.sort_values('cluster', inplace=True)
    data.to_csv(os.path.join(out_dir, file), sep='\t', index=False)
