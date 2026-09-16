"""Prompt bytes for OCC-1. The QUESTION block and answer rules are byte-identical across arms;
only the carrier (text reference vs rendered frames) and its one-paragraph description differ.

Wording adapted from @oh-my-pi/snapcompact research/prompts/{qa-text,qa-image-multi}.md (MIT).
"""

from __future__ import annotations

import base64
import hashlib
import json

RULES = (
    "- Give short extractive answers: a word or phrase copied from the text.\n"
    "- If you cannot find or cannot read the information, reply exactly UNREADABLE for that question.\n"
    "- Output a numbered list, one answer per line, no commentary."
)

TEXT_HEAD = "Below is reference material. Questions follow in the next block.\n\n<reference>\n"
TEXT_TAIL = "\n</reference>\n\nAnswer the questions using ONLY the reference material above.\n" + RULES

IMAGE_HEAD = (
    "The attached {k} image(s) contain encyclopedia passages rendered as dense bitmaps: monospace "
    "pixel font, {cols} characters per row, up to {rows} rows per image. The text flows "
    "continuously across the images: read each image left-to-right, top-to-bottom, then continue "
    "with the next image in order (image 1 first, image {k} last). Original paragraph breaks were "
    "collapsed to spaces."
)
IMAGE_TAIL = "Answer the questions using ONLY text you can read in the images.\n" + RULES


def question_block(questions: list[dict]) -> str:
    return "Questions:\n" + "\n".join(f"{i + 1}. {q['q']}" for i, q in enumerate(questions))


def text_messages(context: str, questions: list[dict]) -> list[dict]:
    content = [
        {"type": "text", "text": TEXT_HEAD + context + TEXT_TAIL},
        {"type": "text", "text": question_block(questions)},
    ]
    return [{"role": "user", "content": content}]


def image_messages(pngs: list[bytes], cols: int, rows: int, questions: list[dict]) -> list[dict]:
    content: list[dict] = [{"type": "text", "text": IMAGE_HEAD.format(k=len(pngs), cols=cols, rows=rows)}]
    for png in pngs:
        url = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": url}})
    content.append({"type": "text", "text": IMAGE_TAIL})
    content.append({"type": "text", "text": question_block(questions)})
    return [{"role": "user", "content": content}]


def fingerprint(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
