"""Flask extensions - initialized once, bound to app in create_app()."""

from flask_cors import CORS
from flask_caching import Cache

cors = CORS()
cache = Cache()
