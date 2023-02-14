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
It is a character level transformer.

Example
-------

In the following example we load an existing model and predict how to continue the music :

```python
from musiclang_predict.sample import ModelLLM
model = ModelLLM.load_model(name='basic', update=True)
samples = model.sample(start='(I % I.M', num_samples=1, temperature=0.8)
print(samples[0])
```
It will displays a musiclang score : "(I % I.M)(\n\tpiano__0=s0, \n\tpiano__1=s2, \n\tpiano__2=s4)+ \n(V['65'] % I.M)( ...

