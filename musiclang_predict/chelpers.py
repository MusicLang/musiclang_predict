import ctypes
import os
from ctypes import c_void_p, c_char_p, c_float, c_ulonglong, c_int, c_char, c_bool

current_file_path = os.path.dirname(os.path.abspath(__file__))


def load_library():
    # Assuming your package structure follows a certain pattern
    lib_name = "librun.so" if os.name != "nt" else "run.dll"
    lib_path = os.path.join(current_file_path, "c", lib_name)
    return ctypes.CDLL(lib_path)

lib = load_library()

# Set up function prototypes correctly
lib.create_transformer.restype = c_void_p
lib.create_transformer.argtypes = [c_char_p]

lib.free_transformer_external.restype = None
lib.free_transformer_external.argtypes = [c_void_p]

lib.run_model.restype = c_void_p  # Corrected to reflect using a transformer pointer
lib.run_model.argtypes = [
    c_void_p, c_bool, c_char_p, c_char, c_float, c_float,
    c_ulonglong, c_int, c_char_p, c_char_p, c_char_p, c_char_p
]

lib.free_returned_string.restype = None
lib.free_returned_string.argtypes = [c_void_p]



# Function to wrap create_transformer
def create_transformer(checkpoint_path):
    trans = lib.create_transformer(checkpoint_path.encode('utf-8'))
    return trans

def free_transformer_external(transformer_ptr):
    lib.free_transformer_external(transformer_ptr)


def run_transformer_model(transformer_ptr, attention_already_generated, tokenizer_path, temperature=1.0, topp=1.0, rng_seed=0, steps=256, prompt=None, post_prompt=None, mode="generate", system_prompt=None, stop_char="_"):
    # Convert string parameters to bytes, if they are not None and are of type str
    if prompt is not None and isinstance(prompt, str):
        prompt = prompt.encode('utf-8')
    if  post_prompt is not None and isinstance(post_prompt, str):
        post_prompt = post_prompt.encode('utf-8')
    if mode is not None and isinstance(mode, str):
        mode = mode.encode('utf-8')
    if system_prompt is not None and isinstance(system_prompt, str):
        system_prompt = system_prompt.encode('utf-8')
    stop_char = stop_char.encode('utf-8')[0] if stop_char is not None and isinstance(stop_char, str) else ord(" ")

    # Call the modified C function with attention_already_generated
    result_ptr = lib.run_model(
        transformer_ptr, attention_already_generated, tokenizer_path.encode('utf-8'), stop_char, temperature, topp, rng_seed,
        steps, prompt, post_prompt, mode, system_prompt
    )

    # Convert the result to a Python string, free the returned string
    result_str = ctypes.cast(result_ptr, ctypes.c_char_p).value.decode('utf-8')
    lib.free_returned_string(result_ptr)

    return result_str



def run_for_n_bars(n_bars, checkpoint_path, tokenizer_path, temperature=1.0, topp=1.0, rng_seed=0, steps=256,
                          prompt=None, mode="generate", system_prompt=None,
                          stop_char="_"):
    transformer_ptr = create_transformer(checkpoint_path)

    for i in range(n_bars):
        generated_text = run_transformer_model(
            transformer_ptr=transformer_ptr,
            checkpoint_path=checkpoint_path,
            tokenizer_path=tokenizer_path,
            temperature=temperature,
            topp=topp,
            rng_seed=rng_seed,
            steps=steps,
            prompt=prompt,
            mode=mode,
            system_prompt=system_prompt,
            stop_char=stop_char
        )
        print(generated_text)
        prompt = generated_text

    free_transformer_external(transformer_ptr)
    return prompt


