# Бейзлайны проекта
В директории представлены 4 бейзлайна:
1. Первый смысл для каждой леммы
2. Случайный из двух смыслов для каждой леммы
3. Кластеризация статических эмбеддингов алгоритмом BIRCH
4. Метод, основанный на подходе jamsic на соревановании RUSSE'2018
5. KMeans кластеризация контекстных векторов модели BERT
6. Egvi на подготовленных sense inventories
7. BERT + KMeans с числом значений слова от Egvi

# Запуск
Для установки пакетов:
```pip install -r requirements.txt```

Для запуска выполнить:
``python main.py [наименование метода] [путь до датасета] --model [путь до модели]``

Пример:
`` jamsic datasets/wiki-wiki/train.csv --model models/word2vec/model_wiki.bin --visualize``

Доступные методы:
- birch
- jamsic
- naive (все наивные бейзлайны, для них указывать модель не нужно)
- bert (при выборе этого бейзлайна укажите в аргументе model название модели в huggingface)

Аргумент --visualize необязательный. В случае добавления сохраняет визуализацию итоговой кластеризации.

# Описание файлов и директорий:
- datasets: загруженные датасеты из RUSSE'2018
- scripts: реализации бейзлайнов
- visualizations: визуализации кластеризаций бейзлайнов, основанных на векторах
- main.py: файл для запуска бейзлайнов
- requirements.txt: модули, необходимые для работы

Установить модули можно командой:
    pip install -r requirements.txt

# Оценка бейзлайнов
Для оценки WSI-бейзлайнов используем метрику [ARI](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html)
| Модель | Метод | Средний ARI | Слово | ARI |
|--------|-------|-------------|-------|-----|
| ruscorpora_upos_cbow_300_20_2019| Birch | 0.69 | замок | 0.19 |
| ruscorpora_upos_cbow_300_20_2019| Birch | 0.69 | лук | 0.89 |
| ruscorpora_upos_cbow_300_20_2019| Birch | 0.69 | бор | 1.0 |
| ruscorpora_upos_cbow_300_20_2019| Jamsic | 0.3 | замок | 0.12 |
| ruscorpora_upos_cbow_300_20_2019| Jamsic | 0.3 | лук | 0.56 |
| ruscorpora_upos_cbow_300_20_2019| Jamsic | 0.3 | бор | 0.21 |
| ruwikiruscorpora_upos_skipgram_300_2_2019 | Birch | 0.93 | замок | 0.79 |
| ruwikiruscorpora_upos_skipgram_300_2_2019 | Birch | 0.93 | лук | 1.0 |
| ruwikiruscorpora_upos_skipgram_300_2_2019 | Birch | 0.93 | бор | 1.0 |
| ruwikiruscorpora_upos_skipgram_300_2_2019 | Jamsic | 0.41 | замок | -0.02 |
| ruwikiruscorpora_upos_skipgram_300_2_2019 | Jamsic | 0.41 | лук | 0.92 |
| ruwikiruscorpora_upos_skipgram_300_2_2019 | Jamsic | 0.41 | бор | 0.32 |
| ruwikiruscorpora-func_upos_skipgram_300_5_2019 | Birch | 0.93 | замок | 0.79 |
| ruwikiruscorpora-func_upos_skipgram_300_5_2019 | Birch | 0.93 | лук | 1.0 |
| ruwikiruscorpora-func_upos_skipgram_300_5_2019 | Birch | 0.93 | бор | 1.0 |
| ruwikiruscorpora-func_upos_skipgram_300_5_2019 | Jamsic | 0.58 | замок | -0.08 |
| ruwikiruscorpora-func_upos_skipgram_300_5_2019 | Jamsic | 0.58 | лук | 0.82 |
| ruwikiruscorpora-func_upos_skipgram_300_5_2019 | Jamsic | 0.58 | бор | 1.0 |
| news_upos_skipgram_300_5_2019 | Birch | 0.54 | замок | 0.02 |
| news_upos_skipgram_300_5_2019 | Birch | 0.54 | лук | 0.92 |
| news_upos_skipgram_300_5_2019 | Birch | 0.54 | бор | 0.66 |
| news_upos_skipgram_300_5_2019 | Jamsic | 0.15 | замок | 0.11 |
| news_upos_skipgram_300_5_2019 | Jamsic | 0.15 | лук | 0.42 |
| news_upos_skipgram_300_5_2019 | Jamsic | 0.15 | бор | -0.07 |
| tayga_upos_skipgram_300_2_2019 | Birch | 0.67 | замок | 0.02 |
| tayga_upos_skipgram_300_2_2019 | Birch | 0.67 | лук | 1.0 |
| tayga_upos_skipgram_300_2_2019 | Birch | 0.67 | бор | 1.0 |
| tayga_upos_skipgram_300_2_2019 | Jamsic | 0.36 | замок | 0.1 |
| tayga_upos_skipgram_300_2_2019 | Jamsic | 0.36 | лук | 0.75 |
| tayga_upos_skipgram_300_2_2019 | Jamsic | 0.36 | бор | 0.24 |
| tayga-func_upos_skipgram_300_5_2019 | Birch | 0.81 | замок | 0.43 |
| tayga-func_upos_skipgram_300_5_2019 | Birch | 0.81 | лук | 1.0 |
| tayga-func_upos_skipgram_300_5_2019 | Birch | 0.81 | бор | 1.0 |
| tayga-func_upos_skipgram_300_5_2019 | Jamsic | 0.47 | замок | 0.05 |
| tayga-func_upos_skipgram_300_5_2019 | Jamsic | 0.47 | лук | 0.65 |
| tayga-func_upos_skipgram_300_5_2019 | Jamsic | 0.47 | бор | 0.71 |
| ruwikiruscorpora_upos_cbow_300_10_2021 | Birch | 0.43 | замок | 0.1 |
| ruwikiruscorpora_upos_cbow_300_10_2021 | Birch | 0.43 | лук | 0.4 |
| ruwikiruscorpora_upos_cbow_300_10_2021 | Birch | 0.43 | бор | 0.79 |
| ruwikiruscorpora_upos_cbow_300_10_2021 | Jamsic | 0.42 | замок | 0.17 |
| ruwikiruscorpora_upos_cbow_300_10_2021 | Jamsic | 0.42 | лук | 0.45 |
| ruwikiruscorpora_upos_cbow_300_10_2021 | Jamsic | 0.42 | бор | 0.63 |
| sberbank-ai/sbert_large_nlu_ru | BERT KMeans | 0.85 | замок | 0.94 |
| sberbank-ai/sbert_large_nlu_ru | BERT KMeans | 0.85 | лук | 0.82 |
| sberbank-ai/sbert_large_nlu_ru | BERT KMeans | 0.85 | бор | 0.79 |
| ruwikiruscorpora-func_upos_skipgram_300_5_2019 | EGVI | 0.59 | замок | 0.82 |
| ruwikiruscorpora-func_upos_skipgram_300_5_2019 | EGVI | 0.59 | лук | 0.43 |
| ruwikiruscorpora-func_upos_skipgram_300_5_2019 | EGVI | 0.59 | бор | 0.51 |
| sberbank-ai/sbert_large_nlu_ru | EGVI + BERT | 0.64 | замок | 0.96 |
| sberbank-ai/sbert_large_nlu_ru | EGVI + BERT | 0.64 | лук | 0.73 |
| sberbank-ai/sbert_large_nlu_ru | EGVI + BERT | 0.64 | бор | 0.22 |

Для WSD – ARI и [Accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)

Для оценки метода "Случайный из двух смыслов" вычислены средние метрики из трёх запусков метода

| Метод | Среднее Accuracy | Средний ARI | Слово | Accuracy | ARI |
|-------| ---------------- |-------------|--------|----------|----|
| Первый смысл для каждого |  0.57 | 0.0 | замок | 0.72| 0.0|
| Первый смысл для каждого |  0.57 | 0.0 | лук | 0.59 | 0.0|
| Первый смысл для каждого |  0.57 | 0.0 | суда | 0.74 | 0.0|
| Первый смысл для каждого |  0.57 | 0.0 | бор | 0.25 |0.0 |
| Случайный из двух смыслов | 0.55 | 0.0 |замок | 0.5 | -0.01|
| Случайный из двух смыслов | 0.55 | 0.01 | лук | 0.55| -0.01 |
| Случайный из двух смыслов | 0.55 | 0.01 | суда| 0.57| 0.01 |
| Случайный из двух смыслов | 0.55 | 0.01 | бор | 0.59| 0.01 |
