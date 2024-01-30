MusicLang Predict
=================

![MusicLang logo](https://github.com/MusicLang/musiclang/blob/main/documentation/images/MusicLang.png?raw=true "MusicLang")


MusicLang Predict is a tool to create original midi soundtracks with generative AI model.
It can be used for different use cases :
- Predict a new song from scratch (a fixed number of bars)
- Continue a song from a prompt
- Predict a new song from a template (see examples below)
- Continue a song from a prompt and a template

To solve template generation use cases,
we provide an interface to create a template from an existing midi file.

Our transformers models are hosted on Hugging Face and are available here : [MusicLang](https://huggingface.co/MusicLang).

We are based on the MusicLang music language. See : [MusicLang](https://github.com/MusicLang/musiclang) for more information.

Installation
------------

Install the musiclang-predict package with pip :

```bash
pip install musiclang-predict
```


How to use ? 
------------

1. Create a new 8 bars song from scratch :

```python
from musiclang_predict import predict, MusicLangTokenizer
from transformers import GPT2LMHeadModel

# Load model and tokenizer
model = GPT2LMHeadModel.from_pretrained('musiclang/musiclang-4k')
tokenizer = MusicLangTokenizer('musiclang/musiclang-4k')
soundtrack = predict(model, tokenizer, chord_duration=4, nb_chords=8)
soundtrack.to_midi('song.mid', tempo=120, time_signature=(4, 4))
```

2. Or use an existing midi song as a song structure template :
```python
from musiclang_predict import midi_file_to_template, predict_with_template, MusicLangTokenizer
from transformers import GPT2LMHeadModel

# Load model and tokenizer
model = GPT2LMHeadModel.from_pretrained('musiclang/musiclang-4k')
tokenizer = MusicLangTokenizer('musiclang/musiclang-4k')

template = midi_file_to_template('my_song.mid')
soundtrack = predict_with_template(template, model, tokenizer)
soundtrack.to_midi('song.mid', tempo=template['tempo'], time_signature=template['time_signature'])
```

See : [MusicLang templates](https://discovered-scabiosa-ea3.notion.site/Create-a-song-template-with-MusicLang-dfd8cad0a14b464fb3475c7fa19c1a82)
For a full description of our template format.
It's only a dictionary containing information for each chord of the song and some metadata like tempo.
You can even create your own without using a base midi file !

3. Or even use a prompt and a template to create a song

```python
from musiclang_predict import midi_file_to_template, predict_with_template, MusicLangTokenizer
from transformers import GPT2LMHeadModel
from musiclang import Score

# Load model and tokenizer
model = GPT2LMHeadModel.from_pretrained('musiclang/musiclang-4k')
tokenizer = MusicLangTokenizer('musiclang/musiclang-4k')
template = midi_file_to_template('my_song.mid')
# Take the first chord of the template as a prompt
prompt = Score.from_midi('my_prompt.mid', chord_range=(0, 4))
soundtrack = predict_with_template(template, model, tokenizer, 
                                   prompt=prompt,  # Prompt the model with a musiclang score
                                   prompt_included_in_template=True  # To say the prompt score is included in the template
                                   )
soundtrack.to_midi('song.mid', tempo=template['tempo'], time_signature=template['time_signature'])
```

4. Chord prediction with a transformer model

```python
from musiclang_predict import predict_chords, MusicLangTokenizer
from transformers import GPT2LMHeadModel
from musiclang.library import *

prompt = (I % I.M) + (V % I.M)['6'].o(-1)

# Load model and tokenizer
model = GPT2LMHeadModel.from_pretrained('musiclang/musiclang-chord-v2-4k')
tokenizer = MusicLangTokenizer('musiclang/musiclang-chord-v2-4k')
soundtrack = predict_chords(model, tokenizer, chord_duration=4, nb_chords=2, prompt=prompt)

# Give the chord a simple voicing (closed position chord)
soundtrack = soundtrack(b0, b1, b2, b3)

# Save it to midi
soundtrack.to_midi('song.mid', tempo=120, time_signature=(4, 4))

```

Contact us
----------

If you want to help shape the future of open source music generation,
please contact [us](mailto:fgardin.pro@gmail.com)

License
-------

The MusicLang predict package (this package) and its associated models is licensed under the GPL-3.0 License.
The MusicLang base language (musiclang package) is licensed under the BSD 3-Clause License.
