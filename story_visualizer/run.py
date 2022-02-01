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
chapter_1_chunk_2 = chunks[1][0]
visualize_chunk(
    chunk=chapter_1_chunk_2,
    image_folder="alice_chapter1_chunk2",
    text_body_hash=str(hash(text_body)),
)
