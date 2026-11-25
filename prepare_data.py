import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from pypdf import PdfReader
from pypdf.generic import ByteStringObject, ContentStream, NameObject, TextStringObject

PDF_SHA256 = "629f4b668ff43abf2a6e621aefd0f50e165ce8e88c52c05769149e1267d033ec"
GLYPHS = {i: bytes([i]).decode("cp1251", errors="replace") for i in range(128, 256)}
GLYPHS.update({i: " " for i in range(32) if i not in (9, 10)})
GLYPHS.update(dict.fromkeys((0, 1, 8), ""))
GLYPHS.update(
    {
        2: "(",
        3: ")",
        12: "|",
        16: "(",
        17: ")",
        18: "(",
        19: ")",
        20: "[",
        21: "-",
        22: "—",
        26: "{",
        27: "}",
        30: "(",
    }
)
RULE = b"q cm d J w m l S Q".split()
OPEN, CLOSE = "\ue000", "\ue001"
LETTER, SYMBOL = r"[А-Яа-яЁё]", r"[A-Za-zα-ωΑ-Ω∂0-9]"


def read_rows(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_rows(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            print(json.dumps(row, ensure_ascii=False), file=f)


def mark_expression(ops, start, fonts, font, size, target, layers):
    """
    Вставляет метки OPEN, CLOSE вокруг выражения, ширина которого совпадает с длиной черты. При совпадении ширины в ops добавляются соответствующие метки.

    Args: 
        ops: Команды страницы PDF в виде пар (аргументы, оператор)
        start: Индекс первой операции вывода текста под чертой.
        fonts: Словарь шрифтов страницы с таблицами ширин глифов.
        font: Ключ текущего шрифта в словаре fonts.
        size: Размер шрифта.
        target: Длина найденной черты.
        layers: Число надчёркиваний над выражением.
    """
    width = 0
    for args, op in ops[start:]:
        if op == b"Tf" and args[1] == size:
            font = args[0]
            continue
        if op != b"TJ" or not fonts[font]["/BaseFont"].endswith(("+CMMI10", "+CMR10")):
            return
        items = args[0]
        for i, item in enumerate(items):
            if not isinstance(item, bytes):
                width -= float(item) * size / 1000
                continue
            for n, char in enumerate(item, 1):
                if (
                    char
                    not in b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ()+-"
                ):
                    return
                glyph_width = fonts[font]["/Widths"][char - fonts[font]["/FirstChar"]]
                width += glyph_width * size / 1000
                if abs(width - target) <= 0.06 * size: # считаем, что на данном глифе заканчивается черта 
                    items[i : i + 1] = [
                        ByteStringObject(item[:n]),
                        TextStringObject(CLOSE * layers),
                        ByteStringObject(item[n:]),
                    ]
                    ops[start][0][0].insert(0, TextStringObject(OPEN * layers))
                    return
                if width > target:
                    return


def page_text(page):
    """
    Извлекает и очищает текст страницы учебника.

    Args: 
        page: Страница PDF.
    Returns:
        Строка с очищенным текстом и восстановленными надчёркиваниями в LaTeX.
    """
    stream = ContentStream(page["/Contents"], page.pdf, "bytes")
    ops = stream.operations
    codes = [op for args, op in ops]
    fonts = page["/Resources"]["/Font"]
    for i, op in enumerate(codes):
        if op != b"BT" or codes[i - len(RULE) : i] != RULE:
            continue
        transform = ops[i - 8][0]
        width, dy = ops[i - 3][0]
        if transform[:4] != [1, 0, 0, 1] or ops[i - 4][0] != [0, 0] or dy != 0:
            continue
        j = i + 1
        while codes[j : j + 1] == [b"Tf"]:
            j += 1
        if j == i + 1 or codes[j : j + 2] != [b"Td", b"TJ"]:
            continue
        font, size = ops[j - 1][0]
        x, y = ops[j][0]
        if (
            not fonts[font]["/BaseFont"].endswith("+CMMI10")
            or abs(x - transform[4]) > 0.01
        ):
            continue
        height = (transform[5] - y) / size
        if not any(
            lo <= height <= hi for lo, hi in ((0.56, 0.58), (0.81, 0.84), (0.88, 0.90))
        ):
            continue
        layers = 1
        prev = i - len(RULE)
        while codes[prev - len(RULE) : prev] == RULE:
            other = ops[prev - 8][0]
            if other[:5] != transform[:5] or ops[prev - 3][0] != ops[i - 3][0]:
                break
            if abs(other[5] - transform[5] - 0.2 * layers * size) > 0.01:
                break
            layers += 1
            prev -= len(RULE)
        mark_expression(ops, j + 1, fonts, font, size, width, layers)
    page[NameObject("/Contents")] = stream
    text = page.extract_text().translate(GLYPHS)
    text = re.sub(
        r"^(?:ЛЕКЦИИ ПО ТФКП\s*\d+|\d+\s*В\. В\. ГОРЯЙНОВ, Е\. С\. ПОЛОВИНКИН)\s*\n",
        "",
        text,
    )
    text = re.sub(r"[§џ]\s*(\d+)\.", r"§ \1.", text)
    text = re.sub(
        rf"(?<={LETTER})(?={SYMBOL}|{OPEN})|(?<={SYMBOL}|{CLOSE})(?={LETTER})",
        " ",
        text,
    )
    text = re.sub(r"-\n(?=[а-яё])", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return (
        unicodedata.normalize("NFC", text)
        .replace(OPEN, r"\overline{")
        .replace(CLOSE, "}")
    )


def prepare(pdf, output):
    """
    Собирает корпус из PDF и подготовить вопросы и эталоны для dev и own.

    Args:
        pdf: Путь к PDF учебника.
        output: Папка для корпуса, вопросов и эталонов в формате JSONL.

    Returns:
        None.
    """
    output.mkdir(parents=True, exist_ok=True)
    blocks = []
    section = "Содержание"
    for number, page in enumerate(PdfReader(pdf).pages, 1):
        text = page_text(page)
        headings = re.findall(r"§ (\d+)\.\s*([^\n]+)", text)
        if headings:
            n, title = headings[-1]
            section = f"§ {n}. {title.strip()}"
        blocks.append(
            dict(block_id=f"pdf_p{number:04d}", page=number, section=section, text=text)
        )
    write_rows(output / "corpus.jsonl", blocks)

    page_ids = {b["page"]: b["block_id"] for b in blocks}
    bank = read_rows(Path(__file__).with_name("question_bank.jsonl"))
    for split in ("dev", "own"):
        questions, gold = [], []
        for item in (row for row in bank if row["split"] == split):
            questions.append({key: item[key] for key in ("id", "question", "options")})
            ref = {
                key: item[key] for key in ("id", "category", "answer_ids", "rationale")
            }
            pages = [page_ids[n] for n in item["pages"]]
            gold.append(
                {
                    **ref,
                    "abstain": not ref["answer_ids"],
                    "evidence_sets": [pages] if ref["answer_ids"] else [],
                }
            )
        write_rows(output / f"{split}_questions.jsonl", questions)
        write_rows(output / f"{split}_gold.jsonl", gold)


def summarize(gold_path, directory):
    """
    Оценивает прогоны на dev и записывает метрики в experiments.csv.

    Args:
        gold_path: Путь к файлу с эталонами dev-вопросов.
        directory: Папка с ответами и трейсами прогонов и csv.
    """
    gold = {row["id"]: row for row in read_rows(gold_path)}
    table = []
    for mode in ("closed", "lexical-direct", "dev", "without-dense", "without-scalar"):
        answers = {a["id"]: a for a in read_rows(directory / f"{mode}_answers.jsonl")}
        trace = {t["id"]: t for t in read_rows(directory / f"{mode}_trace.jsonl")}
        if gold.keys() != answers.keys() or gold.keys() != trace.keys():
            raise ValueError(f"{mode}: разные наборы ID")
        exact, groups = [], defaultdict(list)
        for qid, g in gold.items():
            a = answers[qid]
            correct = (
                trace[qid]["status"] == "ok"
                and a["abstain"] == g["abstain"]
                and set(a["answer_ids"]) == set(g["answer_ids"])
            )
            exact.append(correct)
            score = float(correct)
            if correct and not g["abstain"] and mode != "closed":
                cited = set(a["citations"])
                f1 = max(
                    2 * len(cited & set(ref)) / (len(cited) + len(ref))
                    for ref in g["evidence_sets"]
                )
                score = 0.8 + 0.2 * f1
            groups[g["category"]].append(score)
        tokens_in = mean(t["input_tokens"] for t in trace.values())
        tokens_out = mean(t["output_tokens"] for t in trace.values())
        calls = mean(t["llm_calls"] for t in trace.values())
        table.append(
            dict(
                name="final" if mode == "dev" else mode.replace("-", "_"),
                answer_accuracy=f"{mean(exact):.6f}",
                Q="" if mode == "closed" else f"{mean(map(mean, groups.values())):.6f}",
                mean_input_tokens=f"{tokens_in:.1f}",
                mean_output_tokens=f"{tokens_out:.1f}",
                mean_llm_calls=f"{calls:.6f}",
                mean_cost=f"{tokens_in + 4*tokens_out + 256*calls:.1f}",
                median_latency_ms=f"{median(t['elapsed_ms'] for t in trace.values()):.1f}",
            )
        )
    with (directory / "experiments.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=table[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--pdf", type=Path, help="PDF для подготовки корпуса и вопросов"
    )
    source.add_argument(
        "--gold", type=Path, help="Эталоны dev для подсчёта experiments.csv"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.pdf:
        prepare(args.pdf, args.output)
    else:
        summarize(args.gold, args.output)
