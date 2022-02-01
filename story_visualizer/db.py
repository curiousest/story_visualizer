from settings import DB_PATH, IMAGES_ROOT
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class WordImage(Base):
    __tablename__ = "wordimage"

    id = Column(Integer, primary_key=True)
    word = Column(String(150), nullable=False)
    part_of_speech = Column(String(250), nullable=True)
    image_path = Column(String(250), nullable=False)

    def get_path(self):
        return f"{IMAGES_ROOT}/source/{self.image_path}"


class VisualizedChunk(Base):
    __tablename__ = "visualizedchunk"

    id = Column(Integer, primary_key=True)
    chunk_hash = Column(String(50), nullable=False)
    # text chunk + word images paths
    inputs_hash = Column(String(50), nullable=False)
    text_body_hash = Column(String(50), nullable=False)
    intermediary_image_path = Column(String(250), nullable=True)
    image_path = Column(String(250), nullable=False)
    visualizer_version = Column(String(10), nullable=False)


# Create an engine that stores data in the local directory's
# sqlalchemy_example.db file.
engine = create_engine(f"sqlite:///{DB_PATH}")
