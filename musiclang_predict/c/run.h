// run.h
#ifndef RUN_H
#define RUN_H

#include <stdlib.h>
#include <sys/types.h>

typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
} Config;

typedef struct {
    float* token_embedding_table;
    float* rms_att_weight;
    float* rms_ffn_weight;
    float* wq;
    float* wk;
    float* wv;
    float* wo;
    float* w1;
    float* w2;
    float* w3;
    float* rms_final_weight;
    float* wcls;
} TransformerWeights;

typedef struct {
    float *x;
    float *xb;
    float *xb2;
    float *hb;
    float *hb2;
    float *q;
    float *k;
    float *v;
    float *att;
    float *logits;
    float* key_cache;
    float* value_cache;
} RunState;

typedef struct {
    Config config;
    TransformerWeights weights;
    RunState state;
    int fd;
    float* data;
    ssize_t file_size;
} Transformer;

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512];
} Tokenizer;

typedef struct {
    float prob;
    int index;
} ProbIndex;

typedef struct {
    int vocab_size;
    ProbIndex* probindex;
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

void malloc_run_state(RunState* s, Config* p);
void free_run_state(RunState* s);
void memory_map_weights(TransformerWeights *w, Config* p, float* ptr, int shared_weights);
void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights, int* fd, float** data, ssize_t* file_size);
void build_transformer(Transformer *t, char* checkpoint_path);
void free_transformer(Transformer* t);
void rmsnorm(float* o, float* x, float* weight, int size);
void softmax(float* x, int size);
void matmul(float* xout, float* x, float* w, int n, int d);
float* forward(Transformer* transformer, int token, int pos);
void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size);
void free_tokenizer(Tokenizer* t);
char* decode(Tokenizer* t, int prev_token, int token);
void safe_printf(char *piece);
int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size);
void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens);
void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed);
void free_sampler(Sampler* sampler);
//char* generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps, char stop_char, bool attention_already_generated);
void read_stdin(const char* guide, char* buffer, size_t bufsize);
//char* run_model(Transformer* transformer, bool attention_already_generated, char *tokenizer_path, char stop_char, float temperature, float topp, unsigned long long rng_seed, int steps, char *prompt, char *mode, char *system_prompt);
#endif // RUN_H
