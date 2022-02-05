from db import engine, Base
from sqlalchemy_utils import database_exists, create_database


if not database_exists(engine.url):
    create_database(engine.url)
    print("Created DB.")
    Base.metadata.create_all(bind=engine)
    print("Created tables.")
else:
    print("DB already exists.")
