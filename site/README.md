# HTML Publication

The HTML article is generated from BLOG.md, which is assembled from the modular
section sources.

    python -m pip install -r site/requirements.txt
    /data/zuoxin/workspace/TeleFuser/.venv/bin/python scripts/build_site.py

The generated page uses relative paths so it can be served from this repository
without rewriting media URLs. The evaluation videos remain original MP4 files
with native controls and synchronized playback.

## GitHub Pages

The repository root contains `index.html`, which redirects to the English HTML
edition under `site/`. Publish the complete repository tree so references from
`site/` to media under `sections/` remain valid.

In the GitHub repository settings, open **Pages** and select:

- **Source:** Deploy from a branch
- **Branch:** `main`
- **Folder:** `/(root)`

After the first deployment, the project site is available at:

`https://uxito-ada.github.io/telefuser-efficient-inference/`

The Chinese edition is available at:

`https://uxito-ada.github.io/telefuser-efficient-inference/site/index.zh.html`
