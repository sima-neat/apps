import logging
from vectordb import initialize_db, main

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    initialize_db()
    main()