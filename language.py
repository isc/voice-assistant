"""The language of a transcript, by fastText's lid.176 (AD-093): a follow-up in another language than its
conversation's is the room talking, not a request."""

from pathlib import Path
from typing import Optional

import fasttext

HOME = "fr"  # a conversation's language, unless its request after the wake word is surely another
# Below, short French words were judged other languages: « Merci. » Italian at 0.65, « Vas-y. » English at 0.82
_SURE = 0.9
_model = fasttext.load_model(str(Path(__file__).parent / "lid" / "lid.176.ftz"))  # 0.1 s, 26 MB


def language_of(text: str) -> Optional[str]:
    """The text's language (ISO 639-1, `fr`) when fastText is sure of it, else None."""
    (label,), (score,) = _model.predict(text.replace("\n", " "), k=1)
    return label.removeprefix("__label__") if score >= _SURE else None
