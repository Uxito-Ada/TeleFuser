tf-kernel Documentation
=======================

**tf-kernel** is a high-performance CUDA kernel library for TeleFuser, providing optimized GPU operations for transformer and diffusion models.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api/index
   development
   contributing

Features
--------

- **Elementwise Operations**: Activation functions (SiLU, GELU), RMS normalization, rotary positional embedding (RoPE), casting
- **GEMM Operations**: FP8, INT8, and FP4 quantized matrix multiplication
- **Attention Variants**: SageAttention v2/v3, Block Sparse Attention
- **Multi-Architecture Support**: SM80 (Ampere), SM90 (Hopper), SM100+ (Blackwell)

Quick Links
-----------

- :doc:`installation` - Installation instructions
- :doc:`quickstart` - Get started quickly
- :doc:`api/index` - API reference
- :doc:`development` - Development guide
- :doc:`contributing` - Contributing guidelines

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
