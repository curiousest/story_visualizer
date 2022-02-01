from logging import info

import requests
from constants import ImageFilename
from settings import GOOGLE_CUSTOM_SEARCH_API_KEY, GOOGLE_SEARCH_ENGINE_ID, IMAGES_ROOT
from sqlalchemy.orm import Session
from sqlalchemy.orm.exc import MultipleResultsFound, NoResultFound

from db import WordImage, engine


def download_image_for_url(url: str, filename: ImageFilename) -> None:
    image_response = requests.get(
        url,
        headers={
            "User-Agent": "Text Visualizer/0.1 (github.com/curiousest) requests/2.27.1"
        },
    )
    image_response.raise_for_status()
    img_data = image_response.content
    with open(filename, "wb") as handler:
        handler.write(img_data)


def download_image_for_text(text: str) -> ImageFilename:
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
    filename = f'{results["items"][0]["title"]}.jpg'
    path = f"{IMAGES_ROOT}/source/{filename}"
    url = results["items"][0]["link"]
    info(f"Retrieving image for text {text} from {url}")
    try:
        download_image_for_url(url, path)
    except requests.exceptions.HTTPError:
        filename = f'{results["items"][1]["title"]}.jpg'
        path = f"{IMAGES_ROOT}/source/{filename}"
        url = results["items"][1]["link"]
        info(f"Retrieving image for text {text} from {url}")
        download_image_for_url(url, path)
    return filename


def get_image_for_word(word: str, part_of_speech: str) -> WordImage:
    word = "Alice"
    part_of_speech = "N"
    with Session(engine) as session:
        results = session.query(WordImage).filter(
            WordImage.word == word, WordImage.part_of_speech == part_of_speech
        )

    try:
        word_image = results.one()
    except MultipleResultsFound:
        raise MultipleResultsFound(
            f"Multiple results for word {word} with part of speech {part_of_speech}."
        )
    except NoResultFound:
        import pdb

        pdb.set_trace()
        pass
    else:
        return word_image

    filename = download_image_for_text(word)

    with Session(engine) as session:
        word_image = WordImage(
            word=word, part_of_speech=part_of_speech, image_path=filename
        )
        session.add(word_image)
        session.commit()

    return word_image
