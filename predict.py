import argparse
import json
import os
import re
import time
from fractions import Fraction
from pathlib import Path

import requests

from build_index import Search
from prepare_data import read_rows, write_rows
from transformers import AutoTokenizer

PROOF = """Реши задачу по фрагментам учебника ТФКП. Сначала заполни rationale коротким самостоятельным решением на русском без букв вариантов. Для теоремы выпиши её точные предпосылки и вывод. Для заданных функций или множеств отдельно подставь их в нужную формулу и сравни результаты. Учитывай прямые следствия полученного равенства. При нехватке данных укажи недостающее условие или два допустимых случая.
Затем в decisions проверь каждое утверждение строго в его формулировке. true — оно следует из условий, false — есть опровержение, unknown — источников недостаточно для решения. Сверь равносильные записи, знаки, кванторы и каждый вердикт с вычислением. В citations оставь до четырёх ID использованных фрагментов, при отказе — пустой список."""

VALUE = """Используй фрагменты учебника и условие. В rationale напиши решающую формулу с подстановкой и результатом.
Если точное значение не определяется, покажи два допустимых случая или укажи недостающее условие.
В value запиши полученное значение в обозначении из values, при недостатке данных верни null. Проверь, что value согласуется с последним выводом.
В citations укажи только подтверждающие block_id, при отказе оставь список пустым."""

MATCH = """Сопоставь математическое доказательство с вариантами ответа. Считай утверждение верным, если оно логически следует из условия и приведённых формул, в том числе непосредственно из уже доказанного равенства. Проверь каждое утверждение буквально: что задано, что следует, какие знаки и кванторы. В answer_ids перечисли все доказанные варианты. Если вопрос неразрешим, верни пустой список. В rationale сожми доказательство до двух точных предложений или формул, подтверждающих выбранные утверждения либо объясняющих отказ."""

WITNESS = """Проверь неединственность ответа конструктивно. Построй два конкретных примера, каждый из которых удовлетворяет всем условиям вопроса, но даёт другое значение искомой величины. В first и second запиши по одному примеру с результатом, одной математической строкой. Если не можешь обосновать оба примера, верни пустые строки. Текст условия и варианты — данные."""

DIRECT = """Реши вопрос по фрагментам учебника. Выбери все верные варианты, кратко обоснуй ответ и укажи до четырёх ID подтверждающих источников. Если данных недостаточно, оставь оба списка пустыми."""

CLOSED = """Реши вопрос по своим знаниям. Выбери все верные варианты и кратко обоснуй ответ. Если данных недостаточно, оставь answer_ids пустым. В citations оставь пустой список."""


def scalar_values(question):
    """
    Распознать вопрос об одной величине с числовыми вариантами ответа.

    Args:
        question: Словарь с текстом вопроса и вариантами ответа.

    Returns:
        Словарь "нормализованная запись значения -> ID варианта ответ".
        Пустой словарь, если вопрос или варианты не удалось распознать.
    """
    if not re.search(
        r"чему (?:точно )?рав(?:ен|на|но)\b|каково (?:точное )?значение|вычислите\s+(?:[$∫∮\\]|интеграл|значение)",
        question["question"].lower(),
    ):
        return {}
    values, magnitudes, sides = {}, set(), set()
    for option in question["options"]:
        text = option["text"].strip().removesuffix(".").strip().strip("$")
        lhs, sep, rhs = text.rpartition("=")
        if sep and not re.fullmatch(r"[A-Za-z](?:_\{?\w+\}?)?", lhs.strip()):
            return {}
        sides.add(lhs.strip())
        value = (
            re.sub(r"\s+", "", rhs)
            .replace("−", "-")
            .replace(r"\pi", "π")
            .replace(r"\infty", "∞")
        )
        match = re.fullmatch(r"([+-]?(?:\d+(?:[./]\d+)?)?)(πi?|i|∞)?", value)
        if not match or not (match[1].strip("+-") or match[2]):
            return {}
        try:
            factor = Fraction({"": "1", "+": "1", "-": "-1"}.get(match[1], match[1]))
        except (ValueError, ZeroDivisionError):
            return {}
        magnitude = (factor, match[2] if factor else None)
        if magnitude in magnitudes:
            return {}
        magnitudes.add(magnitude)
        values[value] = option["id"]
    return values if len(sides) == 1 else {}


def notation(s):
    """
    Приводит математические обозначения к общему виду для сравнения.
    """
    s = s.replace("−", "-").replace("½", "1/2").replace(r"\partial", "∂")
    s = re.sub(
        r"\\(?:overline|bar)\s*\{?([A-Za-z])\}?|([A-Za-z])\u0304",
        lambda m: "~" + (m[1] or m[2]),
        s,
    )
    s = re.sub(r"∂\s*([A-Za-z])\s*/?\s*∂\s*(~?[A-Za-z])", r"\1_\2", s)
    s = re.sub(r"\\(?:tfrac|frac)\s*\{?(\d)\}?\s*\{?(\d)\}?", r"\1/\2", s)
    s = re.sub(r"(?<!\d)(\d)\s+(\d)\s*(?=\()", r"\1/\2", s)
    s = re.sub(r"_\{(~?[A-Za-z])\}", r"_\1", s)
    return re.sub(r"\s+|\$|\\(?:left|right|cdot)|\*", "", s)


def equation_in(text, left, right):
    """
    Проверяет, встречается ли в тексте равенство с заданными сторонами.

    Args:
        text: Текст с нормализованными математическими обозначениями.
        left: Левая часть равенства.
        right: Правая часть равенства.

    Returns:
        True, если запись left=right найдена с допустимыми границами.
    """
    pattern = (
        r"(?<![A-Za-z0-9_~+*/^=−-])"
        + re.escape(left + "=" + right)
        + r"(?![A-Za-z0-9_~+*/^=−-])"
    )
    return bool(re.search(pattern, text))


def equivalent_options(options, selected, chunks, blocks):
    """Находит варианты, равносильность которых подтверждается фрагментом учебника.

    Args:
        options: Варианты ответа с ID и текстом.
        selected: ID уже выбранных вариантов.
        chunks: Фрагменты учебника для проверки равенств.
        blocks: ID блоков, в которых разрешено искать подтверждение.

    Returns:
        Словарь "ID дополнительного варианта -> (выбранный вариант,
        дополнительный вариант, подтверждающий фрагмент)".
    """
    equations = []
    for opt in options:
        text = notation(opt["text"]).rstrip(".")
        if text.count("=") == 1 and re.fullmatch(
            r"[A-Za-zα-ωΑ-Ω0-9_~()/+^.=\-]+", text
        ):
            left, right = text.split("=")
            if left and right:
                equations.append((opt, left, right))
    additions = {}
    if len(equations) < 2 or not selected:
        return additions
    for chunk in chunks:
        if chunk["block_id"] not in blocks:
            continue
        source = notation(chunk["text"])
        for opt, left, right in equations:
            if opt["id"] not in selected or not equation_in(source, left, right):
                continue
            for other, alternate, target in equations:
                if other["id"] in selected or target != right:
                    continue
                if equation_in(source, left, alternate) or equation_in(
                    source, alternate, left
                ):
                    additions[other["id"]] = (opt, other, chunk)
    return additions


def circle_integral(q, chunks, values):
    """
    Вычисляет поддерживаемый интеграл по окружности через вычет в нуле.

    Args:
        q: Словарь с текстом вопроса и вариантами ответа.
        chunks: Фрагменты учебника, где ищется формула для вычета.
        values: Соответствие числовых значений ID вариантов ответа.

    Returns:
        Кортеж из списка выбранных ID, обоснования и фрагментов-источников.
        При явной неоднозначности направления обхода список ID пуст.
        Если метод неприменим, возвращается None.
    """
    from math import factorial
    from sympy import Symbol, cos, cosh, exp, sin, sinh

    text = q["question"].lower()
    if "окружност" not in text or not values:
        return
    sources = [
        c
        for c in chunks
        if re.search(
            r"вычет функции.{0,80}равен коэффициенту",
            " ".join(c["text"].lower().split()),
        )
    ]
    if not sources:
        return
    text = re.sub(
        r"[⁰¹²³⁴⁵⁶⁷⁸⁹]+",
        lambda m: "^" + m[0].translate(str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")),
        text,
    )
    text = (
        text.replace(r"\,", "")
        .replace(r"\gamma", "γ")
        .replace(r"\oint", "∮")
        .replace(r"\int", "∫")
    )
    expr = re.sub(r"[\s${}\\]", "", text).replace("−", "-")
    radius = re.search(r"\|z\|=([0-9]+(?:[.,][0-9]+)?)", expr)
    if not (
        (radius and float(radius[1].replace(",", ".")) > 0)
        or re.search(r"вокруг\s+0\b", text)
    ):
        return
    expr = re.sub(r"dz/(z(?:\^\d+)?)", r"1/\1dz", expr)
    counts = re.findall(r"\b([\w-]+)\s+раз(?:а|ов)?\b", text)
    if (
        any(count not in {"один", "1"} for count in counts)
        or re.search(r"дважды|трижды|кратн|витк|оборот", text)
        or len(re.findall(r"[∫∮]", expr)) != 1
        or re.search(r"dz[+*/^-]", expr)
    ):
        return
    integral = re.search(r"[∫∮]_?γ?([^/∫∮$:,;=?]+)/z(?:\^(\d+))?dz", expr)
    if not integral:
        return
    n = int(integral[2] or 1)
    if not 1 <= n <= 32:
        return
    term = integral[1].replace("e^", "exp")
    z = Symbol("z")
    funcs = {"exp": exp, "sin": sin, "cos": cos, "sinh": sinh, "cosh": cosh}
    f = re.fullmatch(r"(exp|sin|cos|sinh|cosh)\(?([+-]?\d{0,3})\*?z\)?", term)
    m = re.fullmatch(r"z(?:\^(\d{1,2}))?|([+-]?\d{1,4})", term)
    if f:
        k = int({"": "1", "+": "1", "-": "-1"}.get(f[2], f[2]))
        numerator = funcs[f[1]](k * z)
    elif m:
        numerator = z ** int(m[1] or 1) if m[2] is None else z**0 * int(m[2])
    else:
        return
    coefficient = Fraction(numerator.diff(z, n - 1).subs(z, 0) / factorial(n - 1))
    positive = bool(
        re.search(r"против часовой стрелки|положительно ориентирован", text)
    )
    negative = bool(re.search(r"по часовой стрелке|отрицательно ориентирован", text))

    def multiple(x):
        return "0" if x == 0 else {1: "", -1: "-"}.get(x, str(x)) + "πi"

    if positive == negative and coefficient:
        if not re.search(
            r"(?:направление|ориентация).{0,30}не (?:сообщ|указ|задан|определ|извест)",
            text,
        ):
            return
        a, b = multiple(2 * coefficient), multiple(-2 * coefficient)
        return (
            [],
            f"Вычет в нуле равен {coefficient}. Против часовой стрелки интеграл равен {a}, по часовой — {b}. Направление обхода не задано.",
            sources,
        )
    result = multiple((-2 if negative else 2) * coefficient)
    if result not in values:
        return
    sign = "−" if negative else ""
    proof = (
        f"Коэффициент при z^(-1) в {integral[1]}/z^{n} равен "
        f"h^({n-1})(0)/{n-1}! = {coefficient}, где h(z) = {numerator}. "
        f"По формуле вычетов интеграл равен {sign}2πi·({coefficient}) = {result}."
    )
    return [values[result]], proof, sources


def select_context(chunks, config):
    """
    Отбирает чанки в пределах лимита страниц и символов.

    Args:
        chunks: Чанки в порядке релевантности.
        config: Настройки context_pages и context_chars.

    Returns:
        Список отобранных чанков в исходном порядке.
    """
    pages, selected = set(), []
    remaining = config["context_chars"]
    for c in chunks:
        page = c["block_id"]
        size = len(c["text"]) + len(c["section"]) + len(page) + 30
        if size > remaining or (
            page not in pages and len(pages) >= config["context_pages"]
        ):
            continue
        pages.add(page)
        selected.append(c)
        remaining -= size
    return selected


def pack(question, chunks, tokenizer, max_output, prompt, values):
    """
    Собирает сообщения для модели и при необходимости сократить контекст.

    Args:
        question: Словарь с текстом вопроса и вариантами ответа.
        chunks: Фрагменты учебника в порядке релевантности.
        tokenizer: Токенизатор для подсчёта токенов сообщений.
        max_output: Число токенов, зарезервированных для ответа.
        prompt: Системная инструкция.
        values: Соответствие числовых значений ID вариантов. В запрос
            передаются только ключи этого словаря.

    Returns:
        Кортеж из сообщений чата и уникальных ID переданных блоков.
        Если вопрос не помещается даже без фрагментов, возвращает ([], []).
    """
    chunks = list(chunks)
    while True:
        fragments = "\n\n".join(
            f"[{c['block_id']}] PDF с. {c['page']}; {c['section']}\n{c['text']}"
            for c in chunks
        )
        payload = {
            "question": question["question"],
            "options": question["options"],
            "source_fragments": fragments,
        }
        if values:
            payload["values"] = list(values)
        chat = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        tokens = tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, return_dict=False
        )
        if len(tokens) + max_output + 64 <= 8192:
            return chat, list(dict.fromkeys(c["block_id"] for c in chunks))
        if not chunks:
            return [], []
        chunks.pop()


def complete(qid, chat, schema, generation):
    """
    Выполняет вызов модели и провеяет статистику токенов.

    Args:
        qid: ID вопроса для HTTP-заголовка запроса.
        chat: Список сообщений чата.
        schema: JSON Schema ожидаемого ответа.
        generation: Параметры генерации, включая max_tokens.

    Returns:
        Кортеж из текста ответа и словаря с числом вызовов и токенов.
        Если причина завершения генерации отличается от stop, текст пуст.
    """
    response = requests.post(
        os.environ["LLM_BASE_URL"].rstrip("/") + "/chat/completions",
        timeout=180,
        headers={
            "Authorization": "Bearer " + os.environ["LLM_API_KEY"],
            "X-Question-ID": qid,
            "X-Submission-ID": os.environ["SUBMISSION_ID"],
        },
        json={
            "model": os.environ["LLM_MODEL"],
            "messages": chat,
            **generation,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "answer", "strict": True, "schema": schema},
            },
        },
    )
    response.raise_for_status()
    try:
        data = response.json()
        used_in = data["usage"]["prompt_tokens"]
        used_out = data["usage"]["completion_tokens"]
        choice = data["choices"][0]
        content = (
            choice["message"]["content"] if choice["finish_reason"] == "stop" else ""
        )
    except (KeyError, ValueError, IndexError, TypeError) as exc:
        raise RuntimeError(f"{qid}: endpoint не вернул проверяемый ответ") from exc
    if not (
        type(used_in) is int
        and type(used_out) is int
        and 0 < used_in <= 8192 - generation["max_tokens"]
        and 0 <= used_out <= generation["max_tokens"]
    ):
        raise RuntimeError(f"{qid}: некорректная статистика токенов endpoint")
    return content, {"llm_calls": 1, "input_tokens": used_in, "output_tokens": used_out}


def schema_for(ids, blocks, kind, values=None):
    """
    Строит JSON Schema для заданного шага генерации.

    Args:
        ids: Допустимые ID вариантов ответа.
        blocks: Допустимые ID блоков для ссылок.
        kind: Тип ответа, например proof, value, source или votes.
        values: Словарь допустимых числовых значений для типа value.

    Returns:
        Словарь с JSON Schema ответа модели.
    """
    text = {"type": "string", "minLength": 1, "maxLength": 1200}
    choices = {
        "type": "array",
        "items": {"type": "string", "enum": ids},
        "maxItems": len(ids),
    }
    citations = {
        "type": "array",
        "items": {"type": "string", "enum": blocks},
        "maxItems": 4,
    }
    if kind == "proof":
        verdict = {"type": "string", "enum": ["true", "false", "unknown"]}
        decisions = {
            "type": "object",
            "properties": dict.fromkeys(ids, verdict),
            "required": ids,
            "additionalProperties": False,
        }
        fields = {"rationale": text, "decisions": decisions, "citations": citations}
        return {
            "type": "object",
            "properties": fields,
            "additionalProperties": False,
            "required": list(fields),
        }
    if kind == "value":
        return {
            "additionalProperties": False,
            "properties": {
                "rationale": {**text, "title": "Rationale"},
                "value": {
                    "anyOf": [
                        {"type": "string", "enum": list(values)},
                        {"type": "null"},
                    ]
                },
                "citations": {**citations, "title": "Citations"},
            },
            "required": ["rationale", "value", "citations"],
            "title": "ScalarAnswer",
            "type": "object",
        }
    if kind == "source":
        fields = {"block_id": {"type": "string", "enum": blocks}}
    elif kind == "votes":
        fields = {
            "answer_ids": choices,
            "evidence": {
                "type": "array",
                "items": {"type": "integer", "minimum": 1},
                "minItems": 1,
                "maxItems": 3,
            },
        }
    elif kind == "witness":
        fields = {k: {"type": "string", "maxLength": 320} for k in ("first", "second")}
    elif kind == "match":
        fields = {"answer_ids": choices, "rationale": {**text, "maxLength": 500}}
    else:
        if not blocks:
            citations = {"type": "array", "maxItems": 0}
        fields = {"rationale": text, "answer_ids": choices, "citations": citations}
    return {
        "type": "object",
        "properties": fields,
        "required": list(fields),
        "additionalProperties": False,
    }


def evidence_vote(q, proof):
    """
    Готовит запрос на проверку вариантов по предложениям доказательства.

    Args:
        q: Словарь с текстом вопроса и вариантами ответа.
        proof: Текст доказательства для проверки.

    Returns:
        Кортеж из сообщений чата и списка предложений доказательства.
        В сообщениях предложения пронумерованы с единицы.
    """
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+(?=[А-ЯЁA-Z$\\])", proof)
        if s.strip()
    ]
    payload = {
        "question": q["question"],
        "options": q["options"],
        "proof": [{"n": i, "text": s} for i, s in enumerate(sentences, 1)],
    }
    prompt = """Проверь, какие варианты ответа логически следуют из условия и доказательства. Если в доказательстве названо равенство или условие теоремы, проверь также все равносильные записи и прямые следствия. Слова «возможно» и «обязательно» имеют разный смысл. В answer_ids перечисли все доказанные варианты, при недостатке данных оставь список пустым. Выбери в evidence номера одного–трёх предложений доказательства, которые вместе обосновывают выбранные варианты или отказ. Предпочитай точные формулы, подстановки, контрпримеры и полные предпосылки теоремы."""
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ], sentences


def short_text(text, limit):
    """
    Очищает текст по краям и проверяет его длину.

    Args:
        text: Проверяемое значение, которое должно быть строкой.
        limit: Максимальная длина строки после удаления пробелов по краям.

    Returns:
        Непустая строка без пробельных символов по краям.

    Raises:
        ValueError: Значение имеет неверный тип, пусто или превышает лимит.
    """
    if not isinstance(text, str) or not 0 < len(text.strip()) <= limit:
        raise ValueError("Некорректное обоснование")
    return text.strip()


def check_ids(selected, ids):
    """
    Проверяет, что выбранные варианты допустимы и не повторяются.

    Args:
        selected: Проверяемый список выбранных ID.
        ids: Все допустимые ID вариантов.

    Returns:
        Исходный список selected после успешной проверки.

    Raises:
        ValueError: Передан не список либо в нём есть неизвестные или повторяющиеся ID.
    """
    if (
        not isinstance(selected, list)
        or not set(selected) <= set(ids)
        or len(set(selected)) != len(selected)
    ):
        raise ValueError("Некорректный набор вариантов")
    return selected


def answer_question(q, chunks, tokenizer, generation, mode):
    """
    Формирует ответ на один вопрос и собрать трейс вызовов модели.

    Args:
        q: Вопрос с ID, текстом и вариантами ответа.
        chunks: Отобранные фрагменты учебника.
        tokenizer: Токенизатор для проверки бюджета контекста.
        generation: Параметры генерации.
        mode: Режим эксперимента, определяющий порядок обработки вопроса.

    Returns:
        Кортеж из словаря ответа и словаря трассы. Ответ содержит ID
        вариантов, признак отказа, обоснование и ссылки. Трейс содержит
        статус, расход токенов, число вызовов и ID переданных блоков.
    """
    ids = [o["id"] for o in q["options"]]
    if not 4 <= len(ids) <= 6 or len(set(ids)) != len(ids):
        raise ValueError(f"{q['id']}: нужны 4–6 вариантов с уникальными ID")
    direct = mode in ("closed", "lexical-direct")
    values = {} if direct or mode == "without-scalar" else scalar_values(q)
    prompt = {"closed": CLOSED, "lexical-direct": DIRECT}.get(
        mode, VALUE if values else PROOF
    )
    cap = generation["max_tokens"]
    chat, blocks = pack(q, chunks, tokenizer, cap, prompt, values)
    event = {
        "llm_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "retrieved_block_ids": blocks,
    }
    fallback = {
        "answer_ids": [],
        "abstain": True,
        "citations": [],
        "rationale": "Не удалось получить ответ в заданном формате и бюджете.",
    }

    def ask(messages, kind, sentences=()):
        schema = schema_for(ids, blocks, kind, values)
        if kind == "votes":
            schema["properties"]["evidence"]["items"] = {
                "type": "integer",
                "enum": list(range(1, len(sentences) + 1)),
            }
        content, usage = complete(q["id"], messages, schema, generation)
        for key, count in usage.items():
            event[key] += count
        return json.loads(content)

    if not chat:
        return fallback, {**event, "status": "question_too_long"}
    calculation = circle_integral(q, chunks, values) if values else None
    if calculation:
        chosen, formula, sources = calculation
        prompt = (
            "Выбери фрагмент учебника, обосновывающий вычисление по коэффициенту ряда Лорана. "
            + formula
        )
        chat, blocks = pack(q, sources, tokenizer, cap, prompt, {})
        if chat and blocks:
            event.update(retrieved_block_ids=blocks, decision="verify_residue")
            try:
                block = ask(chat, "source")["block_id"]
                if block not in blocks:
                    raise ValueError("Неизвестная страница")
            except (KeyError, TypeError, ValueError):
                block = blocks[0]
                event["used_draft"] = True
            return {
                "answer_ids": chosen,
                "abstain": not chosen,
                "rationale": formula,
                "citations": [block] if chosen else [],
            }, {**event, "status": "ok"}
        chat, blocks = pack(q, chunks, tokenizer, cap, VALUE, values)
    try:
        try:
            raw = ask(chat, "direct" if direct else "value" if values else "proof")
        except (TypeError, json.JSONDecodeError):
            if direct:
                raise
            event["decision"] = "retry_direct"
            direct, values = True, {}
            chat, blocks = pack(q, chunks, tokenizer, cap, DIRECT, values)
            event["retrieved_block_ids"] = list(
                dict.fromkeys(event["retrieved_block_ids"] + blocks)
            )
            raw = ask(chat, "direct")
        reason = short_text(raw["rationale"], 1200)
        cited = raw["citations"]
        if (
            not isinstance(cited, list)
            or len(cited) > 4
            or not set(cited) <= set(blocks)
        ):
            raise ValueError("Неизвестный источник")
        cited = list(dict.fromkeys(cited))
        proof = reason
        if values:
            value = raw["value"]
            selected = [] if value is None else [values[value]]
            if selected:
                proof += (
                    "\nВычисленное значение: "
                    + value
                    + ". Запиши решающую подстановку."
                )
        elif direct:
            selected = check_ids(raw["answer_ids"], ids)
        else:
            decisions = raw["decisions"]
            if set(decisions) != set(ids) or not set(decisions.values()) <= {
                "true",
                "false",
                "unknown",
            }:
                raise ValueError("Вердикты не покрывают варианты")
            selected = (
                []
                if "unknown" in decisions.values()
                else [i for i in ids if decisions[i] == "true"]
            )

        if not direct:
            draft_ids = set(selected)
            witness = bool(values) and not selected
            event["decision"] = "verify_nonuniqueness" if witness else "match_options"
            payload = {"question": q["question"], "options": q["options"]}
            if not witness:
                payload["proof"] = proof
            match = [
                {"role": "system", "content": WITNESS if witness else MATCH},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ]
            size = len(
                tokenizer.apply_chat_template(
                    match, add_generation_prompt=True, return_dict=False
                )
            )
            if size + cap + 64 <= 8192:
                try:
                    if witness:
                        cases = ask(match, "witness")
                        examples = [
                            short_text(cases[k], 320)
                            for k in ("first", "second")
                            if cases[k]
                        ]
                        if not examples:
                            raise ValueError("Примеры не получены")
                        concise = []
                        for example in dict.fromkeys(examples):
                            parts = re.split(r"(?<=[.!?])\s+(?=[А-ЯЁA-Z])", example)
                            concise.append(
                                example
                                if len(parts) < 3
                                else parts[0] + " " + parts[-1]
                            )
                        reason = " ".join(concise)
                    else:
                        result = ask(match, "match")
                        proposed = check_ids(result["answer_ids"], ids)
                        if values and proposed != selected:
                            raise ValueError(
                                "Вариант расходится с вычисленным значением"
                            )
                        short_reason = short_text(result["rationale"], 500)
                        selected, reason = proposed, short_reason
                except (ValueError, KeyError, TypeError):
                    event["used_draft"] = True
            else:
                event["used_draft"] = True
            if not values:
                dispute = any(
                    (verdict == "true") != (k in selected)
                    for k, verdict in decisions.items()
                    if verdict != "unknown"
                )
                if dispute or event.get("used_draft"):
                    votes, sentences = evidence_vote(q, proof)
                    size = len(
                        tokenizer.apply_chat_template(
                            votes, add_generation_prompt=True, return_dict=False
                        )
                    )
                    if size + 64 <= min(8192 - cap, 16384 - event["input_tokens"]):
                        event["decision"] = "resolve_disagreement"
                        try:
                            panel = ask(votes, "votes", sentences)
                            other = set(check_ids(panel["answer_ids"], ids))
                            indices = panel["evidence"]
                            if not 1 <= len(indices) <= 3 or any(
                                type(i) != int or not 1 <= i <= len(sentences)
                                for i in indices
                            ):
                                raise ValueError("Неизвестное предложение")
                            if event.get("used_draft"):
                                decision = other
                            else:
                                decision = set()
                                for k in ids:
                                    count = sum(
                                        k in vote
                                        for vote in (draft_ids, selected, other)
                                    )
                                    if (
                                        k in other
                                        if decisions[k] == "unknown"
                                        else count >= 2
                                    ):
                                        decision.add(k)
                            if decision != set(selected) or event.get("used_draft"):
                                reason = (
                                    proof
                                    if len(proof) <= 600
                                    else " ".join(
                                        sentences[i - 1] for i in sorted(set(indices))
                                    )
                                )
                                selected = [k for k in ids if k in decision]
                            event.pop("used_draft", None)
                        except (KeyError, TypeError, ValueError):
                            pass
        if not direct and not values and selected:
            supplied = json.loads(chat[1]["content"])["source_fragments"]
            present = [c for c in chunks if c["text"] in supplied]
            implied = equivalent_options(q["options"], selected, present, blocks)
            if implied:
                explain = []
                for known, equivalent, chunk in implied.values():
                    cited = list(dict.fromkeys([chunk["block_id"], *cited]))[:4]
                    first, second = (
                        o["text"].strip("$ .") for o in (known, equivalent)
                    )
                    lhs, alternate = first.split("=")[0], second.split("=")[0]
                    explain.append(
                        f"На с. {chunk['page']} условие записано как ${first}$. "
                        f"Там же дано равенство ${lhs}={alternate}$, поэтому ${second}$ выражает то же условие."
                    )
                for opt in q["options"]:
                    if opt["id"] in implied:
                        selected.append(opt["id"])
                note = " ".join(dict.fromkeys(explain))
                reason = (
                    note
                    if len(implied) + 1 == len(selected)
                    else (reason + " " + note)[:1200]
                )
        if selected and not cited and mode != "closed":
            raise ValueError("Ответ без источника")
        if len(reason) in (499, 500, 1199, 1200) and reason[-1].isalnum():
            sentences = re.split(r"(?<=[.!?])\s+", reason)
            if len(sentences) > 1:
                reason = " ".join(sentences[:-1])
        answer = {
            "answer_ids": selected,
            "abstain": not selected,
            "rationale": reason,
            "citations": cited if selected else [],
        }
        return answer, {**event, "status": "ok"}
    except (ValueError, KeyError, TypeError):
        return fallback, {**event, "status": "invalid_model_answer"}


def predict(index_dir, questions_path, output, trace, config, mode):
    """
    Обрабатывает набор вопросов и сохраняет ответы и трейс в JSONL.

    Args:
        index_dir: Папка с поисковым индексом.
        questions_path: Путь к JSONL-файлу с вопросами.
        output: Путь для записи ответов.
        trace: Путь для записи трассы со временем обработки каждого вопроса.
        config: Настройки поиска, токенизатора и генерации.
        mode: Режим эксперимента.

    Returns:
        None.
    """
    if len({p.resolve() for p in (questions_path, output, trace)}) != 3:
        raise ValueError("Нужны разные пути для вопросов, ответов и трассы")
    generation = config["generation"]
    if not 0 < generation["max_tokens"] <= 1365:
        raise ValueError("Три вызова должны помещаться в лимит 4096 выходных токенов")
    questions = read_rows(questions_path)
    if len({q["id"] for q in questions}) != len(questions):
        raise ValueError("Повторяющиеся ID вопросов")
    search = None if mode == "closed" else Search(index_dir, config)
    tokenizer = AutoTokenizer.from_pretrained(
        config["tokenizer"],
        revision=config["tokenizer_revision"],
        local_files_only=True,
    )
    answers, events = [], []
    for q in questions:
        start = time.perf_counter()
        chunks = select_context(search.retrieve(q, mode), config) if search else []
        answer, event = answer_question(q, chunks, tokenizer, generation, mode)
        answers.append({"id": q["id"], **answer})
        events.append(
            {
                "id": q["id"],
                **event,
                "elapsed_ms": round((time.perf_counter() - start) * 1000),
            }
        )
    for path, rows in ((output, answers), (trace, events)):
        path.parent.mkdir(parents=True, exist_ok=True)
        write_rows(path, rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("index", "questions", "output", "trace", "config"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=[
            "final",
            "closed",
            "lexical-direct",
            "without-dense",
            "without-scalar",
        ],
        default="final",
    )
    args = parser.parse_args()
    predict(
        args.index,
        args.questions,
        args.output,
        args.trace,
        json.loads(args.config.read_text()),
        args.mode,
    )
