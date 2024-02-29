MusicLang : Controllable Symbolic Music Generation
========================================================

![MusicLang logo](https://github.com/MusicLang/musiclang/blob/main/documentation/images/MusicLang.png?raw=true "MusicLang")


🎶  <b>&nbsp; You want to generate music that you can export to your favourite DAW in MIDI ?</b>


🎛️ <b>&nbsp; You want to control the chord progression of the generated music ? </b>


🚀  <b>&nbsp; You need to run it fast on your laptop without a gpu ?</b>


Here is MusicLang Predict, your controllable music copilot.

I just want to try !
--------------------

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MA2mek826c05BjbWk2nRkVv2rW7kIU_S#scrollTo=rUc7BCCn5wjl)

Go to our Colab, we have a lot of cool examples. From generating creative musical ideas to continuing a song with a specified chord progression.

I am more serious about it
--------------------------

Install the musiclang-predict package :

```bash
!pip install musiclang_predict
```

Then open your favourite notebook and start generating music in a few lines :

```python
from musiclang_predict import MusicLangPredictor
nb_tokens = 1024 
temperature = 0.9  # Don't go over 1.0, at your own risks !
top_p = 1.0 # <=1.0, Usually 1 best to get not too much repetitive music
seed = 16  # change here to change result, or set to 0 to unset seed

ml = MusicLangPredictor('musiclang/musiclang-v2') # Only available model for now

score = ml.predict(
    nb_tokens=nb_tokens,  # 1024 tokens ~ 25s of music (depending of the number of instruments generated)
    temperature=temperature,
    topp=top_p,
    rng_seed=seed # change here to change result, or set to 0 to unset seed
)
score.to_midi('test.mid') # Open that file in your favourite DAW, score editor or even in VLC
```

You were talking about controlling the chord progression ?
----------------------------------------------------------

You had a specific harmony in mind am I right ?
That's why we allow a fine control over the chord progression of the generated music.
Just specify it as a string like below, choose a time signature and let the magic happen.

```python
from musiclang_predict import MusicLangPredictor

# Control the chord progression
# Chord qualities available : M, m, 7, m7b5, sus2, sus4, m7, M7, dim, dim0.
# You can also specify the bass if it belongs to the chord (eg : Bm/D)
chord_progression = "Am CM Dm E7 Am" # 1 chord = 1 bar
time_signature = (4, 4) # 4/4 time signature, don't be too crazy here 
nb_tokens = 1024 
temperature = 0.8
top_p = 1.0
seed = 42

ml = MusicLangPredictor('musiclang/musiclang-v2')

score = ml.predict_chords(
    chord_progression,
    time_signature=time_signature,
    temperature=temperature,
    topp=top_p,
    rng_seed=seed # set to 0 to unset seed
)
score.to_midi('test.mid', tempo=120, time_signature=(4, 4))
```

Disclaimer : The chord progression is not guaranteed to be exactly the same as the one you specified. It's a generative model after all.
Usually it will happen when you use an exotic chord progression and if you set a high temperature.


That's cool but I have my music to plug in ...
------------------------------------------------

Don't worry, we got you covered. You can use your music as a template to generate new music.
Let's continue some Bach music with a chord progression he could have used : 
```python
from musiclang_predict import MusicLangPredictor
from musiclang_predict import corpus

song_name = 'bach_847' # corpus.list_corpus() to get the list of available songs
chord_progression = "Cm C7/E Fm F#dim G7 Cm"
nb_tokens = 1024 
temperature = 0.8 
top_p = 1.0 
seed = 3666 

ml = MusicLangPredictor('musiclang/musiclang-v2')

score = ml.predict_chords(
    chord_progression,
    score=corpus.get_midi_path_from_corpus(song_name),
    time_signature=(4, 4),
    nb_tokens=1024,
    prompt_chord_range=(0,4),
    temperature=temperature,
    topp=top_p,
    rng_seed=seed # set to 0 to unset seed
)

score.to_midi('test.mid', tempo=110, time_signature=(4, 4))
```

What's coming next ?
---------------------

We are working on a lot of cool features, some are already encoded in the model :
- A control over the instruments used in each bar and their properties (note density, pitch range, average velocity)
- Some performances improvements over the inference C script
- A faster distilled model for real-time generation that can be embedded in plugins or mobile applications
- An integration into a DAW as a plugin
- Some specialized smaller models depending on our user's needs

How does that work ? 
---------------------

If you want to learn more about how we are moving toward symbolic music generation, go to our [technical blog](https://musiclang.github.io/).
The tokenization, the model are described in great details. 

We are using a LLAMA2 architecture (many thanks to Andrej Karpathy awesome [llama2.c](https://github.com/karpathy/llama2.c)), trained on a large dataset of midi files (The CC0 licensed [LAKH](https://colinraffel.com/projects/lmd/)).
We heavily rely on preprocessing the midi files to get an enriched tokenization that describe chords & scale for each bar.
The is also helpful for normalizing melodies relative to the current chord/scale.


Contributing & Contact us
-------------------------

We are looking for contributors to help us improve the model, the tokenization, the performances and the documentation.
If you are interested in this project, open an issue, a pull request, or even [contact us directly](https://www.musiclang.io/contact).

License
-------

MusicLang Predict (This package) is licensed under the GPL-3.0 License.
However please note that specific licenses applies to our models. If you would like to use the model in your product, please
[contact us](https://www.musiclang.io/contact). We are looking forward to hearing from you !

The MusicLang base language package on which the model rely ([musiclang package](https://github.com/musiclang/musiclang)) is licensed under the BSD 3-Clause License.
