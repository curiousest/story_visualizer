from typing import Dict, List, Tuple

Paragraph = str
Chapter = List[Paragraph]
Chunk = List[Paragraph]
ChapterChunk = List[Chunk]
CHUNK_SIZE = 2500

WordPartOfSpeech = Tuple[str, str]
TokenizedText = List[WordPartOfSpeech]
WordFrequencies = Dict[WordPartOfSpeech, int]

Subject = str
Object = str
Verb = str


ImageFilename = str
Url = str
