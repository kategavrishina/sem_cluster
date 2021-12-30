# Бейзлайны проекта
В директории представлены 4 бейзлайна:
1. Первый смысл для каждой леммы
2. Случайный из двух смыслов для каждой леммы
3. Кластеризация статических эмбеддингов алгоритмом BIRCH
4. Метод, основанный на подходе jamsic на соревановании RUSSE'2018

Будут добавлены:
1. BERT-бейзлайн
2. Свой смысл для каждой леммы

# Запуск
Для запуска выполнить:
    python main.py [наименование метода] [путь до датасета] --model [путь до модели]

Пример:
    python main.py jamsic datasets/wiki-wiki/train.csv --model models/word2vec/model_wiki.bin

Доступные методы:
- birch
- jamsic
- naive (все наивные бейзлайны)

# Описание файлов и директорий:
- datasets: загруженные датасеты из RUSSE'2018
- scripts: реализации бейзлайнов
- main.py: файл для запуска бейзлайнов
- requirements.txt: модули, необходимые для работы

Установить модули можно командой:
    pip install requirements.txt

# Оценка бейзлайнов
Для оценки WSI-бейзлайнов используем метрику [ARI](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html)

| Метод | ARI |
| --- | ----------- |
| Кластеризация Birch | 0.728 |
| Подход jamsic | 0.455 |

Для WSD – [Accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)

| Метод | ARI |
| --- | ----------- |
| Первый смысл для каждого |  0.635 |
| Случайный из двух смыслов | 0.498 |
