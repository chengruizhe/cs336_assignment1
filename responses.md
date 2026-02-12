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


## adamW accounting
(a) How much peak memory does running AdamW require? Decompose your answer based on the
memory usage of the parameters, activations, gradients, and optimizer state. Express your answer
in terms of the batch_size and the model hyperparameters (vocab_size, context_length,
num_layers, d_model, num_heads). Assume d_ff = 4 ×d_model.
For simplicity, when calculating memory usage of activations, consider only the following compo-
nents:
• Transformer block
– RMSNorm(s)
– Multi-head self-attention sublayer: QKV projections, Q⊤ K matrix multiply, softmax,
weighted sum of values, output projection.
– Position-wise feed-forward: W1 matrix multiply, SiLU, W2 matrix multiply
• final RMSNorm
• output embedding
• cross-entropy on logits

B = batch
S = context length
d = d_model
h = num_heads
L = num_layers
V = vocab_size
d_ff = 4d
T = B * S

### Transformer
Param,Activation,Gradient,Optimizer State
RMSNorm (x2), 2d, 0, 2d, 4d
QKV proj, 3d^2, 4Td, 3d^2, 6d^2
QK, 0, BhS^2, 0, 0
Softmax, 0, BhS^2, 0, 0
V, 0, Td, 0, 0
out proj, d^2, Td, d^2, 2d^2
W1, 4d^2, 4Td, 4d^2, 8d^2
W3, 4d^2, 4Td, 4d^2, 8d^2
silu, 0, 4Td, 0, 0
swiglu, 0, 4Td, 0, 0
W2, 4d^2, Td, 4d^2, 8d^2
Total params = 2d + 3d^2 + d^2 + 4d^2 + 4d^2 + 4d^2 = 16d^2 + 2d
Total activations = 4Td + BhS^2 + BhS^2 + Td + Td + 4Td * 4 + Td = 23Td + 2BhS^2
Total gradient = 2d + 3d^2 + d^2 + 4d^2 + 4d^2 + 4d^2 = 16d^2 + 2d
Total optim = 32d^2 + 4d
Total = 64d^2 + 8d + 23BSd + 2BhS^2

### Final RMSNorm
Total = d + d + 2d = 4d

### Output Embedding
Total = Vd + BSV + Vd + 2Vd = 4Vd + BSV

### CrossEntropy
Total = 0

(b) Instantiate your answer for a GPT-2 XL-shaped model to get an expression that only depends on
the batch_size. What is the maximum batch size you can use and still fit within 80GB memory?
vocab_size : 50,257
context_length : 1,024
num_layers : 48
d_model : 1,600
num_heads : 25
d_ff : 6,400

Total number of floats:
Transformer:
(64 * 1600^2 + 8 * 1600 + 23 * B * 1024 * 1600 + 2 * B * 25 * 1024**2) * 48
= (163.85M + 90.1M * B) * 48
= 7864.8M + 4324.8M * B

Total bytes = (7864.8M + 4324.8M * B + 4 * 1600 + 4 * 50257 * 1600 + B * 1024 * 50257) * 4 bytes
      = (8186.5M + 4376.26M * B) * 4 bytes
      = 32.75 GB + 17.5GB * B
with 80GB of memory, B must be <= 2 to fit.

(c) How many FLOPs does running one step of AdamW take?
Deliverable: An algebraic expression, with a brief justification.
14 * number of params