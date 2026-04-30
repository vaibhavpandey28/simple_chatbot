from service.db import run_query

def get_data_from_db(query: str):
    return run_query(query)