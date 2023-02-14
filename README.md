Musiclang Predict library
==========================

This library is the prediction module from musiclang (https://github.com/MusicLang/musiclang).


Install
--------

```
git clone https://github.com/MusicLang/musiclang_predict
```

Models
------

At the time being there is only one model called "basic" that can be used for your predictions.
- It is a character level transformer.
- It only takes into account piano instrument
- It will be coherent only at very short term

Example
-------

In the following example we load an existing model and predict how to continue the music :

```python
from musiclang_predict.sample import ModelLLM
model = ModelLLM.load_model(name='basic')
samples = model.sample(start='(I % I.M', num_samples=1, temperature=0.8)
print(samples[0])
```
It will displays a musiclang score : "(I % I.M)(\n\tpiano__0=s0, \n\tpiano__1=s2, \n\tpiano__2=s4)+ \n(V['65'] % I.M)( ...


You can also load a model from a folder url : 

```python
from musiclang_predict.sample import ModelLLM
model = ModelLLM.load_model(path='/path/to/my/model')
samples = model.sample(start='(I % I.M', num_samples=1, temperature=0.8)
print(samples[0])
```


Colab example
--------------

Don't hesitate to play with the google colab available here : 
https://colab.research.google.com/drive/1CsmkjTebXxJck3UBeai5XVvFTFrt0ZTL?usp=sharing

You will see how to use it for prediction.