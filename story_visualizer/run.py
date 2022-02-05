import logging

from chunker import BASE_TEXT_FOLDER, get_chunks, get_text
from image_builder import visualize_chunk

logging.basicConfig(level=logging.INFO)


text_body_filename = "alice_in_wonderland_gutenberg.txt"

text_body = get_text(
    f"{BASE_TEXT_FOLDER}/{text_body_filename}",
    "https://www.gutenberg.org/files/11/11-0.txt",
)
logging.info("Splitting text body into chunks")
chunks = get_chunks(text_body)
chapter = 2
chunk = 1
visualize_chunk(
    chunk=chunks[chapter+1][chunk],
    image_folder=f"aliceinwonderland/chapter{chapter}/chunk{chunk}",
    text_body_hash=str(hash(text_body)),
)
