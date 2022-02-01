import datetime
import re
from logging import info
from typing import List

import requests
from constants import CHUNK_SIZE, ChapterChunk, Paragraph

BASE_TEXT_FOLDER = "/Users/Douglas.Hindson/workspace/story_visualizer/text"


def get_text(filename, url=None) -> str:
    try:
        with open(filename, "r") as f:
            return f.read()
    except FileNotFoundError:
        pass
    info(f"Retrieving text from {url}")
    response = requests.get(url)
    text_body = response.content.decode("utf-8")
    with open(filename, "w") as f:
        f.write(text_body)
    with open(filename.replace(".txt", "_metadata.txt"), "w") as f:
        f.write(f"url: {url}\nretrieved: {datetime.datetime.now()}")
    return text_body


def get_chunks(
    text_body: str, chapter_re="CHAPTER .*\.\n.*\n\n\n", paragraph_re="\n\n"
) -> List[ChapterChunk]:
    chapters = re.split(chapter_re, text_body)
    chapter_chunks = []
    for chapter in chapters:
        paragraphs = re.split(paragraph_re, chapter)
        p = 0
        chapter_chunk = []
        chunk: List[Paragraph] = []
        while p < len(paragraphs):
            chunk.append(paragraphs[p])
            if sum(len(c) for c in chunk) >= CHUNK_SIZE:
                chapter_chunk.append(chunk)
                chunk = []
            p += 1
        if chunk:
            chapter_chunk.append(chunk)
        chapter_chunks.append(chapter_chunk)
    return chapter_chunks
