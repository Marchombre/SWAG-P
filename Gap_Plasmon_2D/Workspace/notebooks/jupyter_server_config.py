import os
from traitlets.config import Config

c = Config()

# 1) Indiquer à Tornado où servir les fichiers statiques
c.JupyterServerApp.tornado_settings = {
    "static_path": os.path.join(os.path.dirname(__file__), "static"),
    "static_url_prefix": "/static/"
}

# 2) Dire à Voilà de copier tous vos "static/" sous /voila/static/
c.VoilaConfiguration.static_root       = "static"
c.VoilaConfiguration.static_url_prefix = "/static/"
