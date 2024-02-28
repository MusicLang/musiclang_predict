import ctypes
import os

current_file_path = os.path.dirname(os.path.abspath(__file__))



def load_library():
    # Assuming your package structure follows a certain pattern
    lib_name = "librun.so" if os.name != "nt" else "run.dll"
    lib_path = os.path.join(current_file_path, "c", lib_name)
    return ctypes.CDLL(lib_path)

def run_transformer_model(checkpoint_path, tokenizer_path, temperature=1.0, topp=1.0, rng_seed=0, steps=256,
                          prompt=None, mode="generate", system_prompt=None,
                          stop_char="_"
                          ):

    lib = load_library()

    # Define the return type and argument types for run_model
    lib.run_model.restype = ctypes.c_void_p  # Use void pointer for the returned string
    lib.run_model.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char, ctypes.c_float, ctypes.c_float,
        ctypes.c_ulonglong, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p
    ]

    # Define the argument types for free_returned_string
    lib.free_returned_string.restype = None
    lib.free_returned_string.argtypes = [ctypes.c_void_p]

    # Convert string parameters to bytes, if they are not None
    if prompt is not None:
        prompt = prompt.encode('utf-8')
    if mode is not None:
        mode = mode.encode('utf-8')
    if system_prompt is not None:
        system_prompt = system_prompt.encode('utf-8')
    if stop_char is not None:
        stop_char = stop_char.encode('utf-8')
    if stop_char is None:
        stop_char = " ".encode('utf-8')
    # Call the function
    result_ptr = lib.run_model(
        checkpoint_path.encode('utf-8'), tokenizer_path.encode('utf-8'), stop_char, temperature, topp, rng_seed,
        steps, prompt, mode, system_prompt
    )

    # Convert the result to a Python string
    result_str = ctypes.cast(result_ptr, ctypes.c_char_p).value.decode('utf-8')

    # Free the returned string after use
    lib.free_returned_string(result_ptr)

    return result_str


def run_for_n_bars(n_bars, checkpoint_path, tokenizer_path, temperature=1.0, topp=1.0, rng_seed=0, steps=256,
                          prompt=None, mode="generate", system_prompt=None,
                          stop_char="_"):

    for i in range(n_bars):
        generated_text = run_transformer_model(
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
    return prompt


