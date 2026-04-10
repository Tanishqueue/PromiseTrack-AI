
"""
run.py
Single entrypoint for the PromiseTrack AI Flask app.
"""
 
import config
from app import create_app
 
app = create_app()
 
if __name__ == "__main__":
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG,
    )
 