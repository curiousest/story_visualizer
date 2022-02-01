import os
from logging import info
from typing import Callable, Optional

import git
from constants import Chunk
from image_retriever import get_image_for_word
from PIL.Image import Image, open
from settings import IMAGES_ROOT
from summarizer import get_summary

from db import VisualizedChunk

TopLeft = lambda fg, bg: (0, 0)
TopRight = lambda fg, bg: (bg.size[0] - fg.size[0], 0)
BottomRight = lambda fg, bg: (bg.size[0] - fg.size[0], bg.size[1] - fg.size[1])
BottomLeft = lambda fg, bg: (0, bg.size[1] - fg.size[1])


def apply_foreground(foreground: Image, background: Image, location: Callable) -> Image:
    if foreground.size[0] > foreground.size[1]:
        new_width = min(background.size[0] // 2, foreground.size[0])
        new_height = round(foreground.size[1] * (new_width / foreground.size[0]))
    else:
        new_height = min(background.size[1] // 2, foreground.size[1])
        new_width = round(foreground.size[0] * (new_height / foreground.size[1]))
    foreground = foreground.resize((new_width, new_height))
    foreground = foreground.convert("RGBA")
    background = background.convert("RGBA")
    background.paste(
        foreground,
        location(foreground, background),
        foreground,
    )
    return background


def run_deepdream(
    intermediary_path: str,
    image_path: str,
    target_image: Optional[str] = None,
    n_iter: int = 4,
) -> None:
    target_image_cmd = f"-g {IMAGES_ROOT}/themes/{target_image}" if target_image else ""
    os.system(
        f"../deepdream.sh -i {intermediary_path} -o {image_path} -n {n_iter} {target_image_cmd}"
    )


def visualize_chunk(
    chunk: Chunk, image_folder: str, text_body_hash: str
) -> VisualizedChunk:
    info("Summarizing chunk.")
    subject, verb, obj1, obj2 = get_summary(chunk)

    info("Retrieving images for summary.")
    subject_word_image_path = get_image_for_word(subject, "N").get_path()
    subject_word_image = open(subject_word_image_path)
    verb_word_image_path = get_image_for_word(verb, "V").get_path()
    verb_word_image = open(verb_word_image_path)
    obj1_word_image_path = get_image_for_word(obj1, "N").get_path()
    obj1_word_image = open(obj1_word_image_path)
    obj2_word_image_path = get_image_for_word(obj2, "N").get_path()
    obj2_word_image = open(obj2_word_image_path)

    info("Combining images.")
    result = apply_foreground(subject_word_image, verb_word_image, TopLeft)
    result = apply_foreground(obj1_word_image, result, TopRight)
    result = apply_foreground(obj2_word_image, result, BottomRight)
    intermediary_path = f"images/intermediary/{image_folder}.png"
    result.save(f"{IMAGES_ROOT}/intermediary/{image_folder}.png")

    info("Applying DeepDream.")
    result_folder = f"results/{image_folder}"
    run_deepdream(intermediary_path, result_folder)

    chunk_hash = hash(" ".join(chunk))
    images_hash = hash(
        "".join(
            [
                subject_word_image_path,
                verb_word_image_path,
                obj1_word_image_path,
                obj2_word_image_path,
            ]
        )
    )
    inputs_hash = hash(chunk_hash + images_hash)
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    info(f"Saving results to {image_folder}.")
    visualized_chunk = VisualizedChunk(
        image_path=image_folder,
        intermediary_image_path=f"{image_folder}.png",
        chunk_hash=chunk_hash,
        inputs_hash=inputs_hash,
        text_body_hash=text_body_hash,
        visualizer_version=sha,
    )
    return visualized_chunk
