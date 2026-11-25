import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile
import numpy as np
from sentence_transformers import SentenceTransformer
from prepare_data import read_rows


def encoder(config):
    """
    Загружает энкодер из конфига
    """
    return SentenceTransformer(
        config["encoder"],
        revision=config["encoder_revision"],
        device="cpu",
        local_files_only=True,
    )


def passage(block):
    """
    Подготавливает текст для E5

    Args:
        block: Запись с названием раздела в section и текстом в text.

    Returns:
        Строка с префиксом passage:, названием раздела и текстом.
    """
    return f"passage: {block['section']}\n{block['text']}"


def chunks_of(block, tokenizer, size):
    """
    Разбивает текст блока на чанки, по возможности сохраняя целые предложения.

    Args:
        block: Исходный блок с текстом, названием раздела и метаданными.
        tokenizer: Токенизатор E5, используемый для подсчёта токенов и разрезания
            длинных формул без пробелов.
        size: Бюджет токенов на чанк с учётом заголовка, префикса и служебных токенов.

    Yields:
        Словарь с метаданными исходного блока и текстом одного чанка.
    """
    text = block["text"]
    prefix = f"passage: {block['section']}\n"
    budget = size - len(
        tokenizer.encode(prefix, add_special_tokens=True)
    )  # остальной бюджет для текста
    if budget < 16:
        raise ValueError("Заголовок раздела не помещается в окно E5")

    def fits(part):
        return (
            len(tokenizer.encode(part, add_special_tokens=False, verbose=False))
            <= budget
        )

    ends = [
        m.end() for m in re.finditer(r"\n{2,}|[.!?]\s+(?=[А-ЯЁA-Z])", text)
    ]  # концы предложений
    current = ""
    start = 0
    for end in [*ends, len(text)]:
        sentence = text[start:end]
        start = end
        if fits(current + sentence):
            current += sentence
            continue
        if current:
            yield {**block, "text": current}
        current = ""
        for word in re.findall(
            r"\s*\S+\s*|\s+", sentence
        ):  # итерируемся по словам с пробелами
            if current and not fits(current + word):
                yield {**block, "text": current}
                current = ""
            while not fits(
                word
            ):  # возможно разделяем одно слово (например, формулу) на несколько чанков
                offsets = tokenizer(
                    word,
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                    verbose=False,
                )["offset_mapping"]
                cut = max(1, offsets[budget - 4][0])
                yield {**block, "text": word[:cut]}
                word = word[cut:]
            current += word
    if current:
        yield {**block, "text": current}


def build(corpus, output, config):
    """
    Разбивает корпус на чанки, вычисляет эмбеддинги и сохраняет индекс.

    Args:
        corpus: Путь к JSONL-файлу с исходными блоками.
        output: Папка, в которую записывается index.npz.
        config: Настройки модели, её ревизии и размера чанков.

    """
    model = encoder(config)
    blocks = read_rows(corpus)
    if not blocks or len({b["block_id"] for b in blocks}) != len(
        blocks
    ):  # пустой ли список и есть ли дубликаты блоков
        raise ValueError("Нужен непустой корпус с уникальными block_id")
    size = config["chunk_tokens"]
    if not 32 <= size <= model.max_seq_length:
        raise ValueError("Размер чанка должен помещаться в окно E5")
    chunks = [c for b in blocks for c in chunks_of(b, model.tokenizer, size)]
    vectors = model.encode(
        [passage(c) for c in chunks],   # добавляем к блоку название раздела
        batch_size=16,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    output.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output, delete=False) as file:
        path = Path(file.name)
    try:
        with path.open("wb") as file:
            np.savez(
                file,
                chunks=json.dumps(chunks, ensure_ascii=False),
                vectors=vectors,
                revision=config["encoder_revision"],
                chunk_tokens=size,
            )
        path.replace(output / "index.npz")
    finally:
        path.unlink(missing_ok=True)
    print(f"{len(blocks)} страниц, {len(chunks)} чанков")


def terms(text):
    """
    Разбивает текст на токены для BM25.

    Args:
        text: Текст чанка или поискового запроса.

    Returns:
        Список токенов в нижнем регистре с заменой ё на е.
        Математические операторы и команды LaTeX сохраняются как отдельные токены.
    """
    return re.findall(
        r"\\[a-z]+|[^\W_]+(?:[′'])?|[=<>+−*/^\-]", text.lower().replace("ё", "е")
    )


def math_terms(text):
    """
    Добавляются подсказки "интеграл, Коши, формула, производной" для \int, \oint, ∫, ∮ и "экспонента"

    Args:
        text: Текст вопроса, в котором ищутся записи интеграла и экспоненты.

    Returns:
        Строка с подсказками через пробел или пустая строка без совпадений.
    """
    hints = []
    if re.search(r"\\(?:int|oint)|[∫∮]", text) and re.search(
        r"/\s*[A-Za-zζξ](?:\s*\^\s*\{?\d+|[²³⁴⁵⁶⁷⁸⁹])", text
    ):
        hints.extend(
            ("интеграл", "Коши", "формула", "производной")
        )  # \int, \oint, ∫, ∮ -> добавляется подсказка "интеграл, Коши, формула, производной"
    if re.search(r"e\s*\^\s*\{?z", text):  # e^z или e^{z}
        hints.append("экспонента")
    return " ".join(hints)


class Search:
    """
    Поиск по чанкам учебника с помощью BM25 и эмбеддингов E5.
    """
    def __init__(self, directory, config):
        with np.load(directory / "index.npz", allow_pickle=False) as index:
            if (
                "chunk_tokens" not in index
                or index["chunk_tokens"].item() != config["chunk_tokens"]
                or index["revision"].item() != config["encoder_revision"]
            ):
                raise ValueError("Пересоберите индекс с текущими настройками E5")
            self.chunks = json.loads(index["chunks"].item())
            self.vectors = index["vectors"]
        self.model = encoder(config)
        self.k = config["candidate_k"]
        self.bags = [
            Counter(terms(passage(c))) for c in self.chunks
        ]  # частоты токенов в каждом чанке
        self.df = Counter(
            term for bag in self.bags for term in bag
        )  # число чанков, в которых встречается каждый токен
        lengths = np.array([sum(bag.values()) for bag in self.bags])
        self.norm = 1.5 * (0.25 + 0.75 * lengths / lengths.mean())

    def lexical(self, query):
        """
        Ищет чанки по совпадению токенов с помощью BM25.

        Args:
            query: Текст поискового запроса.

        Returns:
            Список из не более k индексов чанков с положительной оценкой, отсортированный по убыванию оценки BM25.
        """
        scores = np.zeros(len(self.chunks))
        for term in sorted(set(terms(query))):
            df = self.df[term]
            if df:
                tf = np.array([bag[term] for bag in self.bags])
                idf = math.log1p((len(scores) - df + 0.5) / (df + 0.5))
                scores += idf * tf * 2.5 / (tf + self.norm)
        return [
            i for i in np.argsort(-scores, kind="stable")[: self.k] if scores[i] > 0
        ]

    def dense(self, query):
        vector = self.model.encode("query: " + query, normalize_embeddings=True)
        return np.argsort(-(self.vectors @ vector), kind="stable")[
            : self.k
        ]  # индексы k ближайших чанков

    def retrieve(self, question, mode="final"):
        """
        Ищет чанки по косинусному сходству с эмбеддингом запроса.
    
        Args:
            query: Текст запроса без префикса query:.

        Returns:
            Массив с индексами не более k чанков по убыванию сходства.
        """
        stem = question["question"]
        query = "\n".join([stem, *(o["text"] for o in question["options"])])
        lexical = self.lexical(query)
        if mode == "lexical-direct":
            return [self.chunks[i] for i in lexical]
        ranks = [lexical]
        if mode != "without-dense":
            if (
                len(self.model.tokenizer.encode("query: " + query))
                > self.model.max_seq_length
            ):
                query = stem
            ranks.append(self.dense(query))
        hints = math_terms(stem)
        if hints:
            ranks.append(self.lexical(hints))
        scores = Counter()
        for ranking in ranks:
            for rank, i in enumerate(ranking, 1):
                scores[i] += 1 / (60 + rank)  # RRF
        order = sorted(scores, key=lambda i: (-scores[i], i))
        return [self.chunks[i] for i in order[: self.k]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "output", "config"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    build(args.corpus, args.output, json.loads(args.config.read_text()))
