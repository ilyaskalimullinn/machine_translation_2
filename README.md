# Machine Translation 2

Пет-проект по переводу с русского языка на английский. Это вторая версия проекта, в которой были исправлены значительные ошибки [предыдущей версии](https://github.com/ilyaskalimullinn/machine_translation)

Я использовал архитектуру Transformer, а также провел анализ и чистку данных, так что советую просмотреть ноутбуки:

1. `1-data-analysis.ipynb` — просмотр, фильтрация и визуализация данных, обзор на тексты, добавление токенизаторов
2. `2-model.ipynb` — обучение модели трансформера

Модули проекта:

- `download` — модуль для скачивания данных (подробнее про использование читайте в первом ноутбуке)
- `model` — архитектура Transformer со всеми внутренними модулями, такими как MultiHeadAttention и FeedForward
- `train` — модуль, который содержит функции для обучения модели
- `util` — вспомогательный модуль, который содержит для сохранения и загрузки моделей, класс Translator для быстрого перевода с использованием модели и т.д.

При реализации архитектуры я опирался на следующие источники:

- Основная статья — [Attention is All You Need](https://arxiv.org/pdf/1706.03762)
- Статья о том, в какой последовательности применять `Layer Norm`, `Attention` и `FeedForward` — [On Layer Normalization in the Transformer Architecture](https://arxiv.org/pdf/2002.04745)
- Гитхаб репозиторий, к которому я обращался, когда что-то ломалось — [bentrevett](https://github.com/bentrevett/pytorch-seq2seq/tree/main)

Источник данных— корпус текстов [OpenSubtitle](<http://www.opensubtitles.org/>), сами данные — с ресурса [OPUS](<https://opus.nlpl.eu/OpenSubtitles/ru&en/v2018/OpenSubtitles>), данные были использованы в статье P. Lison and J. Tiedemann, 2016, [OpenSubtitles2016: Extracting Large Parallel Corpora from Movie and TV Subtitles](http://www.lrec-conf.org/proceedings/lrec2016/pdf/947_Paper.pdf). In Proceedings of the 10th International Conference on Language Resources and Evaluation (LREC 2016).
