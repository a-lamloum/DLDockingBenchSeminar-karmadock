# KarmaDock course website

This directory contains a fully offline static course companion for the KarmaDock seminar project.
It is generated entirely with the Python 3 standard library; no package installation, network
connection, or web server is required.

## View the site

Open `index.html` directly in a browser, or serve the repository root locally:

```bash
python3 -m http.server 8000
```

Then visit <http://localhost:8000/website/>.

Search, navigation, workflow images, downloads, and the theme toggle work both through the local
server and when `index.html` is opened with a `file://` URL. Links marked with ↗ are external and
require a network connection only if followed. Every page has a complete Arabic counterpart; use the
language link in the header to switch between English and Arabic while staying on the same section.

## Regenerate

From anywhere, run the generator by path:

```bash
python3 -I /absolute/path/to/DLDockingBenchSeminar-karmadock/website/build_site.py
```

From the repository root, the shorter equivalent is:

```bash
python3 -I website/build_site.py
```

The build is deterministic and replaces only the declared generated HTML pages and `assets/`.
It preserves this file, `build_site.py`, and everything under `content/`. English source content is
read from the repository documentation. Authored Arabic translations, localized UI text, translation
status, and quizzes live under `content/` and are never regenerated.

## Generated structure

- Twenty English pages and twenty Arabic counterparts, for 40 generated HTML pages in total.
- Each language includes six core course pages, the walkthrough index, and thirteen source-oriented
  code guides.
- Twelve paired core quiz sources under `content/quizzes/`, rendered at the end of their pages.
- Fourteen paired walkthrough quiz sources covering all 54 content sections in each language,
  rendered immediately after the section they assess. Together with the core quizzes, each language
  contains 60 comprehension blocks.
- `assets/`: local CSS and JavaScript, per-language search indexes, workflow images, and copied downloads.

The site intentionally does not bundle the large model checkpoints or the reference-structure ZIP;
their repository locations are described on the relevant pages.
