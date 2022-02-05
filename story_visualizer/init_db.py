from db import Base, engine
from sqlalchemy_utils import create_database, database_exists

if not database_exists(engine.url):
    create_database(engine.url)
    print("Created DB.")
    Base.metadata.create_all(bind=engine)
    print("Created tables.")
else:
    print("DB already exists.")
