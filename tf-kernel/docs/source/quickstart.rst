Quickstart
==========

This guide will help you get started with tf-kernel quickly.

Installation
------------

First, install tf-kernel:

.. code-block:: bash

   pip install tf-kernel

Basic Usage
-----------

Elementwise Operations
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import torch
   import tf_kernel

   # RMS Normalization
   x = torch.randn(1, 1024, 768, device='cuda')
   weight = torch.randn(768, device='cuda')
   normalized = tf_kernel.rmsnorm(x, weight)

   # SiLU and Mul (SwiGLU)
   x = torch.randn(1, 1024, 768, device='cuda')
   result = tf_kernel.silu_and_mul(x)

GEMM Operations
^^^^^^^^^^^^^^^

.. code-block:: python

   import torch
   import tf_kernel

   # INT8 Quantized Matrix Multiplication
   A = torch.randn(1024, 512, device='cuda')
   B = torch.randn(512, 256, device='cuda')
   A_scale = torch.tensor(1.0, device='cuda')
   B_scale = torch.tensor(1.0, device='cuda')

   C = tf_kernel.int8_scaled_mm(A, B, A_scale, B_scale)

   # FP8 Quantization
   x = torch.randn(1024, 512, device='cuda')
   x_fp8, scale = tf_kernel.tf_per_token_quant_fp8(x)

Attention Operations
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import torch
   import tf_kernel

   # SageAttention v2
   q = torch.randn(1, 8, 1024, 64, device='cuda')
   k = torch.randn(1, 8, 1024, 64, device='cuda')
   v = torch.randn(1, 8, 1024, 64, device='cuda')

   output = tf_kernel.sageattn2_forward(q, k, v)

   # Block Sparse Attention
   # See tf_kernel.block_sparse_attn for detailed usage

Performance Tips
----------------

1. **Use CUDA Graphs**: For repeated operations with the same shapes, consider using CUDA graphs to reduce CPU overhead.

2. **Architecture-Specific Builds**: Build for your specific GPU architecture to get the best performance:

   .. code-block:: bash

      make build-auto

3. **Memory Layout**: Ensure tensors are contiguous when possible for better memory access patterns.

Next Steps
----------

- Check out the :doc:`api/index` for detailed API documentation
- See :doc:`development` for contributing guidelines
- Visit the `GitHub repository <https://github.com/YOUR_ORG/tf-kernel>`_ for more examples
