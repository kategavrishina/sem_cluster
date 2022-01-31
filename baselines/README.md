# Бейзлайны проекта
В директории представлены 4 бейзлайна:
1. Первый смысл для каждой леммы
2. Случайный из двух смыслов для каждой леммы
3. Кластеризация статических эмбеддингов алгоритмом BIRCH
4. Метод, основанный на подходе jamsic на соревановании RUSSE'2018
5. KMeans кластеризация контекстных векторов модели BERT

Будут добавлены:
1. Свой смысл для каждой леммы

# Запуск
Для установки пакетов:
```pip install -r requirements.txt```

Для запуска выполнить:
``python main.py [наименование метода] [путь до датасета] --model [путь до модели]``

Пример:
``python main.py jamsic datasets/wiki-wiki/train.csv --model models/word2vec/model_wiki.bin --visualize``

Доступные методы:
- birch
- jamsic
- naive (все наивные бейзлайны, для них указывать модель не нужно)
- bert (при выборе этого бейзлайна укажите в аргументе model название модели в huggingface)

Аргумент --visualize необязательный. В случае добавления сохраняет визуализацию итоговой кластеризации.

# Описание файлов и директорий:
- datasets: загруженные датасеты из RUSSE'2018
- scripts: реализации бейзлайнов
- main.py: файл для запуска бейзлайнов
- requirements.txt: модули, необходимые для работы

Установить модули можно командой:
    pip install requirements.txt

# Оценка бейзлайнов
Для оценки WSI-бейзлайнов используем метрику [ARI](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html)
| Модель | Метод | Средний ARI | Слово | ARI |
|--------|-------|-------------|-------|-----|
| ruscorpora_upos_cbow_300_20_2019| Birch | 0.57 | замок | 0.2 |
| ruscorpora_upos_cbow_300_20_2019| Birch | 0.57 | лук | 0.82 |
| ruscorpora_upos_cbow_300_20_2019| Birch | 0.57 | суда | 0.24 |
| ruscorpora_upos_cbow_300_20_2019| Birch | 0.57 | бор | 1.0 |
| ruscorpora_upos_cbow_300_20_2019| Jamsic | 0.22 | замок | 0.12 |
| ruscorpora_upos_cbow_300_20_2019| Jamsic | 0.22 | лук | 0.58 |
| ruscorpora_upos_cbow_300_20_2019| Jamsic | 0.22 | суда | -0.01 |
| ruscorpora_upos_cbow_300_20_2019| Jamsic | 0.22 | бор | 0.21 |
| ruwikiruscorpora_upos_skipgram_300_2_2019 | Birch | 0.73 | замок | 0.79 |
| ruwikiruscorpora_upos_skipgram_300_2_2019 | Birch | 0.73 | лук | 0.93 |
| ruwikiruscorpora_upos_skipgram_300_2_2019 | Birch | 0.73 | суда | 0.2 |
| ruwikiruscorpora_upos_skipgram_300_2_2019 | Birch | 0.73 | бор | 1.0 |
| ruwikiruscorpora_upos_skipgram_300_2_2019 | Jamsic | 0.32 | замок | -0.02 |
| ruwikiruscorpora_upos_skipgram_300_2_2019 | Jamsic | 0.32 | лук | 0.86 |
| ruwikiruscorpora_upos_skipgram_300_2_2019 | Jamsic | 0.32 | суда | 0.13 |
| ruwikiruscorpora_upos_skipgram_300_2_2019 | Jamsic | 0.32 | бор | 0.32 |
| ruwikiruscorpora-func_upos_skipgram_300_5_2019 | Birch | 0.75 | замок | 0.79 |
| ruwikiruscorpora-func_upos_skipgram_300_5_2019 | Birch | 0.75 | лук | 1.0 |
| ruwikiruscorpora-func_upos_skipgram_300_5_2019 | Birch | 0.75 | суда | 0.2 |
| ruwikiruscorpora-func_upos_skipgram_300_5_2019 | Birch | 0.75 | бор | 1.0 |
| ruwikiruscorpora-func_upos_skipgram_300_5_2019 | Jamsic | 0.47 | замок | -0.8 |
| ruwikiruscorpora-func_upos_skipgram_300_5_2019 | Jamsic | 0.47 | лук | 0.82 |
| ruwikiruscorpora-func_upos_skipgram_300_5_2019 | Jamsic | 0.47 | суда | 0.15 |
| ruwikiruscorpora-func_upos_skipgram_300_5_2019 | Jamsic | 0.47 | бор | 1.0 |
| news_upos_skipgram_300_5_2019 | Birch | 0.46 | замок | 0.02 |
| news_upos_skipgram_300_5_2019 | Birch | 0.46 | лук | 0.93 |
| news_upos_skipgram_300_5_2019 | Birch | 0.46 | суда | 0.24 |
| news_upos_skipgram_300_5_2019 | Birch | 0.46 | бор | 0.66 |
| news_upos_skipgram_300_5_2019 | Jamsic | 0.15 | замок | 0.12 |
| news_upos_skipgram_300_5_2019 | Jamsic | 0.15 | лук | 0.37 |
| news_upos_skipgram_300_5_2019 | Jamsic | 0.15 | суда | 0.17 |
| news_upos_skipgram_300_5_2019 | Jamsic | 0.15 | бор | -0.07 |
| tayga_upos_skipgram_300_2_2019 | Birch | 0.53 | замок | 0.02 |
| tayga_upos_skipgram_300_2_2019 | Birch | 0.53 | лук | 1.0 |
| tayga_upos_skipgram_300_2_2019 | Birch | 0.53 | суда | 0.09 |
| tayga_upos_skipgram_300_2_2019 | Birch | 0.53 | бор | 1.0 |
| tayga_upos_skipgram_300_2_2019 | Jamsic | 0.48 | замок | 0.09 |
| tayga_upos_skipgram_300_2_2019 | Jamsic | 0.48 | лук | 0.7 |
| tayga_upos_skipgram_300_2_2019 | Jamsic | 0.48 | суда | 0.91 |
| tayga_upos_skipgram_300_2_2019 | Jamsic | 0.48 | бор | 0.24 |
| tayga-func_upos_skipgram_300_5_2019 | Birch | 0.54 | замок | 0.2 |
| tayga-func_upos_skipgram_300_5_2019 | Birch | 0.54 | лук | 0.42 |
| tayga-func_upos_skipgram_300_5_2019 | Birch | 0.54 | суда | 0.24 |
| tayga-func_upos_skipgram_300_5_2019 | Birch | 0.54 | бор | 0.79 |
| tayga-func_upos_skipgram_300_5_2019 | Jamsic | 0.56 | замок | 0.18 |
| tayga-func_upos_skipgram_300_5_2019 | Jamsic | 0.56 | лук | 0.47 |
| tayga-func_upos_skipgram_300_5_2019 | Jamsic | 0.56 | суда | -0.09 |
| tayga-func_upos_skipgram_300_5_2019 | Jamsic | 0.56 | бор | 0.63 |
| ruwikiruscorpora_upos_cbow_300_10_2021 | Birch | 0.41 | замок | 0.2 |
| ruwikiruscorpora_upos_cbow_300_10_2021 | Birch | 0.41 | лук | 0.42 |
| ruwikiruscorpora_upos_cbow_300_10_2021 | Birch | 0.41 | суда | 0.24 |
| ruwikiruscorpora_upos_cbow_300_10_2021 | Birch | 0.41 | бор | 0.79 |
| ruwikiruscorpora_upos_cbow_300_10_2021 | Jamsic | 0.3 | замок | 0.18 |
| ruwikiruscorpora_upos_cbow_300_10_2021 | Jamsic | 0.3 | лук | 0.47 |
| ruwikiruscorpora_upos_cbow_300_10_2021 | Jamsic | 0.3 | суда | -0.09 |
| ruwikiruscorpora_upos_cbow_300_10_2021 | Jamsic | 0.3 | бор | 0.63 |


| Метод | ARI |
| --- | ----------- |
| Кластеризация BERT* | 0.732 |
| Кластеризация Birch | 0.728 |
| Подход jamsic | 0.455 |

\* В качестве бейзлайна BERT использовалась модель sberbank-ai/sbert_large_nlu_ru, алгоритм кластеризации KMeans

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
