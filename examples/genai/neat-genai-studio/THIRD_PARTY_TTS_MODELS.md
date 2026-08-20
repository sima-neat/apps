# Third-party TTS models

Neat GenAI Studio installs and loads server-side voices only from
[`voice_catalog.json`](src/python/ui/voice_catalog.json). Each entry records a
short licence label, pinned upstream repository revision, and SHA-256 checksum
for every downloaded file. A model that is merely present in `assets/` is not
eligible to load. Models under `CC-BY-NC-SA-4.0` are excluded.

This file records attribution and notices; it is not legal advice. Re-check an
upstream model and its training data before adding or updating an allowlist
entry.

## Piper Plus CSS10 (default multilingual/Japanese voice)

- Model: `ayousanz/piper-plus-css10-ja-6lang`
- Model source: <https://huggingface.co/ayousanz/piper-plus-css10-ja-6lang>
- Training dataset: CSS10 Japanese, declared public domain by the model author
- Base model: `ayousanz/piper-plus-base`, licensed CC BY 4.0
- Base-model source: <https://huggingface.co/ayousanz/piper-plus-base>
- Attribution: CSS10 contributors; ayousanz, Piper Plus multilingual base model
- Piper Plus software: MIT, <https://github.com/ayutaz/piper-plus>

The Studio preserves the CC BY 4.0 base-model attribution here and in the
this notice file.

## Piper Plus MERA (optional multilingual voice)

- Model: `kizuna-intelligence/piper-plus-mera-multilingual`
- Model source: <https://huggingface.co/kizuna-intelligence/piper-plus-mera-multilingual>
- Published model licence: Apache License 2.0
- Copyright/model author: Kizuna Intelligence
- Base-model credit: ayousanz, Piper Plus multilingual base model (CC BY 4.0)
- Apache License 2.0 text: <https://www.apache.org/licenses/LICENSE-2.0>

MERA is never part of the default download. Select it in the Studio voice
picker or set `TTS_OPTIONAL_VOICES=mera` to download it from its pinned commit.
The Apache 2.0 and CC BY 4.0 notices above must remain with redistributed model
copies.

## Catalogued dedicated Piper voices

| Catalog id | Language | Published data/model terms | Attribution/source |
| --- | --- | --- | --- |
| `en_US-kristin-medium` | English | Public domain | LibriVox source recordings |
| `en_US-ljspeech-medium` *(optional)* | English | Public domain | LJ Speech dataset |
| `de_DE-thorsten-medium` | German | CC0 1.0 | Thorsten Voice dataset |
| `es_ES-davefx-medium` | Spanish | CC0 1.0 | Nabu Casa voice dataset |
| `fr_FR-siwis-medium` | French | CC BY 4.0 | SIWIS French Speech Synthesis Database |
| `it_IT-paola-medium` | Italian | CC0 1.0 | Paola Italian voice dataset |
| `no_NO-talesyntese-medium` | Norwegian | CC0 1.0 | Norwegian Talesyntese dataset |
| `pt_BR-faber-medium` | Portuguese | CC0 1.0 | Nabu Casa voice dataset |
| `vi_VN-vais1000-medium` | Vietnamese | CC BY 4.0 | VAIS-1000 corpus |
| `zh_CN-huayan-medium` | Chinese | Unknown | HuaYan TTS |
| `zh_CN-chaowen-medium` *(optional)* | Chinese | Mixed | CC0 voice dataset; fine-tuned from Xiao Ya |

The exact repository, commit, and file checksums are in the catalog.
`piper-tts` 1.7.0 is a separate GPL-3.0 server runtime; that software licence is
distinct from each voice/model-data licence.

## Explicit exclusions

The server catalog excludes Hi-Fi-Captain Japanese, KSS Korean, and HFC English
because they use `CC-BY-NC-SA-4.0`. Korean therefore uses browser TTS when the
client has a Korean voice, or a text-only response otherwise.
