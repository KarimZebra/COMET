# Copyright (C) 2020 Unbabel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
r"""
Transformers v4/v5 compatibility shims
======================================
    transformers v5 collapses the historical slow/fast tokenizer split:
    the ``*TokenizerFast`` classes are no longer exported, only the
    consolidated ``*Tokenizer`` class. The ``*Compat`` aliases below
    resolve to the Fast variant on v4 and to the consolidated class on
    v5, so encoders can stay version-agnostic.

    The ``*Compat`` suffix is intentional: in transformers v4, names
    like ``BertTokenizer`` refer to the *slow* Python tokenizer (still
    exported alongside ``BertTokenizerFast``). Using a distinct alias
    avoids confusion with that class.
"""
try:
    from transformers import BertTokenizerFast as BertTokenizerCompat
except ImportError:
    from transformers import BertTokenizer as BertTokenizerCompat  # type: ignore[assignment]

try:
    from transformers import XLMRobertaTokenizerFast as XLMRobertaTokenizerCompat
except ImportError:
    from transformers import XLMRobertaTokenizer as XLMRobertaTokenizerCompat  # type: ignore[assignment]

try:
    from transformers import RemBertTokenizerFast as RemBertTokenizerCompat
except ImportError:
    from transformers import RemBertTokenizer as RemBertTokenizerCompat  # type: ignore[assignment]


__all__ = [
    "BertTokenizerCompat",
    "XLMRobertaTokenizerCompat",
    "RemBertTokenizerCompat",
]
