## Transformer Accounting
(a) Consider GPT-2 XL, which has the following configuration:
vocab_size : 50,257
context_length : 1,024
num_layers : 48
d_model : 1,600
num_heads : 25
d_ff : 6,400
Suppose we constructed our model using this configuration. How many trainable parameters
would our model have? Assuming each parameter is represented using single-precision floating
point, how much memory is required to just load this model?

Parms:
Embedding: 
    vocab_size x d_model = 50257 x 1600 = 80.4M
rope:
    0
transformers:
    2 * rms_norm + multi-head attention + ffn
= 2 * d_model + d_model * d_model * (3 + 1) + d_model * d_ff * 3
= 2 * 1600 + 1600 * 1600 * 4 + 1600 * 6400 * 3
= 3200 + 10.24M + 30.72M
= 40.96M
40.96M * 48 = 1966M

final rms norm:
    d_model = 1600

final lm head
    d_model * vocab_size
= 1600 * 50257
= 80.4M
Total:
1966 + 80.4 * 2 = 2.1268 B params
