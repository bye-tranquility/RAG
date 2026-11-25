# Доказательный RAG по ТФКП

## Айкашев Максим Сергеевич, Б05-321

### На чём запускалось?
CPU, Apple M3 16 Gb

### Структура
* `prepare_data.py` извлекает текст из PDF и разделяет исходную разметку `question_bank.jsonl` на вопросы и эталоны. 
* `build_index.py` сохраняет фрагменты и их векторы E5 в один файл NumPy. 
* `predict.py`делает RRF между BM25 и E5, передаёт фрагменты Qwen, которая находит доказательство со ссылками. На вопрос числовой величине Qwen вычисляет её значение или сообщает о неоднозначности. Для простого кругового контура код сверяет интеграл с коэффициентом ряда Лорана в SymPy, если поиск нашёл формулировку теоремы о вычете. Qwen указывает подтверждающий фрагмент. Для остальных задач второй вызов сопоставляет доказательство с ID вариантов или строит контрпримеры. При расхождении вердиктов дополнительная проверка выбирает предложения доказательства. Код объединяет равносильные варианты, когда переданный фрагмент явно содержит соответствующее тождество и исходное равенство. Оборванный первый ответ повторяется прямым запросом. В трейс записываются решения о следующем вызове, расход токенов и фактически переданные страницы.

### Как запустить?

Из корня репозитория запускаем последовательность команд, подготовливающих окружение, скачивающих учебник, загружающих эмбеддер E5 и саму Qwen во временную папку.

```bash
run=$(mktemp -d)
python3.11 -m venv "$run/venv"
source "$run/venv/bin/activate"
export PYTHONDONTWRITEBYTECODE=1
python -m pip install -r requirements.txt
curl -fL 'https://new.mipt.ru/upload/mathematics/study/uchebniki/%D0%9B_%D0%A2%D0%A4%D0%9A%D0%9F_%D0%93%D0%BE%D1%80%D1%8F%D0%B9%D0%BD%D0%BE%D0%B2_%D0%9F%D0%BE%D0%BB%D0%BE%D0%B2%D0%B8%D0%BD%D0%BA%D0%B8%D0%BD_2025.pdf' -o "$run/book.pdf"
python prepare_data.py --pdf "$run/book.pdf" --output "$run/data"
hf download intfloat/multilingual-e5-small --revision 614241f622f53c4eeff9890bdc4f31cfecc418b3 --include 'config.json' 'model.safetensors' 'modules.json' 'sentence_bert_config.json' '1_Pooling/*' 'tokenizer*' 'special_tokens_map.json' 'sentencepiece.bpe.model'
hf download Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --include 'tokenizer*' 'config.json' 'special_tokens_map.json'
export HF_HUB_OFFLINE=1
python build_index.py --corpus "$run/data/corpus.jsonl" --output "$run/index" --config config.json
```

Подготовка проверяет SHA-256 учебника. В `$run/data` появятся `corpus.jsonl`, `dev_questions.jsonl`, `dev_gold.jsonl` и авторский набор. Исходная разметка хранит 15 вопросов из материалов задания и 8 авторских. При сборке номера страниц PDF превращаются в ID блоков.

### Что нужно для получения ответов?

Необходимо задать `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY` и `SUBMISSION_ID` для выданного OpenAI-совместимого endpoint.
В `config.json` записаны параметры измеренного запуска.

Локальный прогон использовал Ollama 0.40.16, Qwen3.5-4B MLX NVFP4, digest `7f3251fa6878a78a606bcb3c074283306e5ad2ea411a7774a9dd98a4bc7f2d13`.

```bash
OLLAMA_HOST=127.0.0.1:11468 OLLAMA_CONTEXT_LENGTH=8192 OLLAMA_NUM_PARALLEL=1 ollama serve
```

При первой загрузке нужно выполнить `OLLAMA_HOST=127.0.0.1:11468 ollama pull qwen3.5:4b` и сравнить digest по ссылке `http://127.0.0.1:11468/api/tags`, причём выполнять обе команды нужно подряд в указанном порядке. При смене порядка Qwen может дать другой ответ на тот же запрос.

```bash
export LLM_BASE_URL=http://127.0.0.1:11468/v1 LLM_MODEL=qwen3.5:4b
export LLM_API_KEY=ollama
for split in dev own; do
  SUBMISSION_ID="local-$split" python predict.py --index "$run/index" \
    --questions "$run/data/${split}_questions.jsonl" --output "$run/${split}_answers.jsonl" \
    --trace "$run/${split}_trace.jsonl" --config config.json
done

for mode in closed lexical-direct without-dense without-scalar; do
  SUBMISSION_ID="local-$mode" python predict.py --mode "$mode" --index "$run/index" \
    --questions "$run/data/dev_questions.jsonl" --output "$run/${mode}_answers.jsonl" \
    --trace "$run/${mode}_trace.jsonl" --config config.json
done
python prepare_data.py --gold "$run/data/dev_gold.jsonl" --output "$run"
```

### Абляции
* `closed` отвечает без учебника
* `lexical-direct` использует один поиск BM25 и один вызов модели
* `without-dense` отключает E5
* `without-scalar` на числовые альтернативы ответы идут как на обычные утверждения

Измерения лежат в [experiments.csv](experiments.csv)
Разбор ошибок находится в [report.md](report.md).
