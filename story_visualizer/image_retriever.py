import datetime
from logging import info
from typing import Tuple

import requests
from constants import ImageFilename, Url
from db import WordImage, engine
from settings import GOOGLE_CUSTOM_SEARCH_API_KEY, GOOGLE_SEARCH_ENGINE_ID, IMAGES_ROOT
from sqlalchemy.orm import Session
from sqlalchemy.orm.exc import MultipleResultsFound, NoResultFound

BASE_SOURCE_FOLDER = f"{IMAGES_ROOT}/source/"


def download_image_for_url(url: str, filename: ImageFilename) -> None:
    path = f"{IMAGES_ROOT}/source/{filename}"
    image_response = requests.get(
        url,
        headers={
            "User-Agent": "Text Visualizer/0.1 (github.com/curiousest/story_visualizer) requests/2.27.1"
        },
    )
    image_response.raise_for_status()
    img_data = image_response.content
    info(f"Saving image to: {path}")
    with open(path, "wb") as handler:
        handler.write(img_data)


def image_filename(word: str) -> str:
    return f"{word}{datetime.datetime.now().isoformat()[:19]}.jpg"


def download_image_for_text(text: str) -> Tuple[ImageFilename, Url]:
    info(f"Searching for image for text {text}")
    response = requests.get(
        "https://www.googleapis.com/customsearch/v1",
        params={
            "q": text,
            "cx": GOOGLE_SEARCH_ENGINE_ID,
            "key": GOOGLE_CUSTOM_SEARCH_API_KEY,
            "fileType": "jpg",
            "count": 2,
            "startIndex": 1,
            "inputEncoding": "utf8",
            "outputEncoding": "utf8",
            "safe": "active",
            "searchType": "image",
            "imgSize": "medium",
        },
    )
    response.raise_for_status()
    results = response.json()
    if len(results["items"]) == 0:
        raise FileNotFoundError
    filename = image_filename(text)
    url = results["items"][0]["link"]
    info(f"Retrieving image for text {text} from {url}")
    try:
        download_image_for_url(url, filename)
    except requests.exceptions.HTTPError:
        url = results["items"][1]["link"]
        info(f"Retrieving image for text {text} from {url}")
        download_image_for_url(url, filename)
    return filename, url


def set_image_for_text(*, url: str, word: str, part_of_speech: str) -> WordImage:
    filename = image_filename(word)
    download_image_for_url(url, filename)
    with Session(engine, expire_on_commit=False) as session:
        word_image = WordImage(
            word=word,
            part_of_speech=part_of_speech,
            image_path=filename,
            source_url=url,
        )
        session.add(word_image)
        session.commit()
    return word_image


def get_image_for_word(word: str, part_of_speech: str) -> WordImage:
    with Session(engine) as session:
        results = session.query(WordImage).filter(
            WordImage.word == word, WordImage.part_of_speech == part_of_speech
        )
        pass

    try:
        word_image = results.one()
    except MultipleResultsFound:
        raise MultipleResultsFound(
            f"Multiple results for word {word} with part of speech {part_of_speech}."
        )
    except NoResultFound:
        pass
    else:
        return word_image

    filename, url = download_image_for_text(word)

    with Session(engine, expire_on_commit=False) as session:
        word_image = WordImage(
            word=word,
            part_of_speech=part_of_speech,
            image_path=filename,
            source_url=url,
        )
        session.add(word_image)
        session.commit()

    return word_image
