![MusicLang logo](https://github.com/MusicLang/musiclang/blob/main/documentation/images/MusicLang.png?raw=true "MusicLang")

<h1 align="center" weight='300' >MusicLang Predict, your controllable music copilot. </h1>

<h4 align="center">
  <a href="https://huggingface.co/musiclang/musiclang-v2"> 🤗 HuggingFace</a> |
  <a href="https://discord.gg/2g7eA5vP">Discord</a> |
  <a href="https://www.linkedin.com/company/musiclang/">Follow us!</a>
</h4>
<br/>

<h4>☞ You want to generate music that you can export to your favourite DAW in MIDI ?</h4> 
<h4>☞ You want to control the chord progression of the generated music ?</h4> 
<h4>☞ You need to run it fast on your laptop without a gpu ?</h4> 

<br/>
<h2 align="center"><b>MusicLang is the contraction of “Music” & “language”: we bring advanced controllability features over music generation by manipulating symbolic music.</b></h2>
<br/>

<summary><kbd>Table of contents</kbd></summary>

- <a href="#quickstart">Quickstart 🚀</a>
    - <a href="#colab">Try it quickly 📙</a>
    - <a href="#install">Install MusicLang ♫</a>
- <a href="#examples">Examples 🎹</a>
    - <a href="#example_1">1. Generate your first music 🕺</a>
    - <a href="#example_2">2. Controlling chord progression generation 🪩</a>
    - <a href="#example_3">3. Generation from an existing music 💃</a>
- <a href="#next">What's coming next at musiclang? 👀</a>
- <a href="#hdw">How does MusicLang work? 🔬</a>
    - <a href="#article_1">1. Annotate chords and scales progression of MIDIs using MusicLang analysis</a>
    - <a href="#article_2">2. The MusicLang tokenizer : Toward controllable symbolic music generation</a>
- <a href="#share">Contributing & spread the word 🤝</a>
- <a href="#license">License ⚖️</a>

<h1 id="quickstart">Quickstart 🚀</h1>
<h2 id="colab">Try it quickly 📙</h2>
<br/>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MA2mek826c05BjbWk2nRkVv2rW7kIU_S?usp=sharing)
[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/musiclang/musiclang-predict)

Go to our <a href="https://colab.research.google.com/drive/1MA2mek826c05BjbWk2nRkVv2rW7kIU_S?usp=sharing">♾Colab</a>, or to our <a href="https://huggingface.co/spaces/musiclang/musiclang-predict">🤗HuggingFace space</a>, we have a lot of cool examples, from generating creative musical ideas to continuing a song with a specified chord progression.
<br/>

<h2 id="install">Install MusicLang ♫</h2> 
<br/>

Install the `musiclang-predict` package :

```bash
pip install musiclang_predict
```
<h2 id="examples">Examples 🎹</h2>

<h3 id="example_1">1. Generate your first music 🕺</h3> 
<br/>

Open your favourite notebook and start generating music in a few lines :

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

<h3 id="example_2">2. Controlling chord progression generation 🪩</h3>
<br/>

You had a specific harmony in mind, right ? MusicLang allows fine control over the chord progression of the generated music.
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

> Disclaimer : The chord progression is not guaranteed to be exactly the same as the one you specified. It's a generative model after all. This may occur more frequently when using an exotic chord progression or setting a high temperature.

<h3 id="example_3">3. Generation from an existing music 💃</h3>
<br/>

What if I want to use MusicLang from an existing music ? Don't worry, we got you covered. You can use your music as a template to generate new music.
Let's continue with some Bach music and explore a chord progression he might have used: 
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

<h2 id="next">What's coming next at MusicLang? 👀</h2>
<br/>

We are working on a lot of cool features, some are already encoded in the model :
- A control over the instruments used in each bar and their properties (note density, pitch range, average velocity);
- Some performances improvements over the inference C script;
- A faster distilled model for real-time generation that can be embedded in plugins or mobile applications;
- An integration into a DAW as a plugin;
- Some specialized smaller models depending on our user's needs;
- And more to come! 😎

<h2 id="hdw">How does MusicLang work? 🔬</h2>
<br/>

If you want to learn more about how we are moving toward symbolic music generation, go to our [technical blog](https://musiclang.github.io/). The tokenization, the model are described in great details: 

<h4 id="article_1"><a href="https://musiclang.github.io/chord_parsing/"> 1. Annotate chords and scales progression of MIDIs using MusicLang analysis</a></h4>
<h4 id="article_2"><a href="https://musiclang.github.io/tokenizer/"> 2. The MusicLang tokenizer : Toward controllable symbolic music generation</a></h4>
<br/> 

We are using a LLAMA2 architecture (many thanks to Andrej Karpathy's awesome [llama2.c](https://github.com/karpathy/llama2.c)), trained on a large dataset of midi files (The CC0 licensed [LAKH](https://colinraffel.com/projects/lmd/)).
We heavily rely on preprocessing the midi files to get an enriched tokenization that describe chords & scale for each bar.
The is also helpful for normalizing melodies relative to the current chord/scale.


<h2 id="share">Contributing & spread the word 🤝</h2> 
<br/>

We are looking for contributors to help us improve the model, the tokenization, the performances and the documentation.
If you are interested in this project, open an issue, a pull request, or even [contact us directly](https://www.musiclang.io/contact).

Whether you're contributing code or just saying hello, we'd love to hear about the work you are creating with MusicLang. Here's how you can reach out to us: 
* Join our [Discord](https://discord.gg/2g7eA5vP) to ask your questions and get support
* Follow us on [Linkedin](https://www.linkedin.com/company/musiclang/)
* Add your star on [GitHub](https://github.com/musiclang/musiclang_predict?tab=readme-ov-file) or [HuggingFace](https://huggingface.co/musiclang/musiclang-4k)

<h2 id="license">License ⚖️</h2>
<br/>

MusicLang Predict (This package) is licensed under the GPL-3.0 License.
However please note that specific licenses applies to our models. If you would like to use the model in your commercial product, please
[contact us](https://www.musiclang.io/contact). We are looking forward to hearing from you !

The MusicLang base language package on which the model rely ([musiclang package](https://github.com/musiclang/musiclang)) is licensed under the BSD 3-Clause License.
