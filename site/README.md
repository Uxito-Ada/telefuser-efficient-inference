# HTML Publication

The HTML article is generated from BLOG.md, which is assembled from the modular
section sources.

    python -m pip install -r site/requirements.txt
    /data/zuoxin/workspace/TeleFuser/.venv/bin/python scripts/build_site.py

The generated page uses relative paths so it can be served from this repository
without rewriting media URLs. The evaluation videos remain original MP4 files
with native controls and synchronized playback.
