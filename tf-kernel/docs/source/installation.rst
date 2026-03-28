Installation
============

Requirements
------------

- Python >= 3.10
- PyTorch == 2.9.1
- CUDA Toolkit 11.8+ or 12.x
- CMake >= 3.26 (for building from source)

Install from PyPI
-----------------

The easiest way to install tf-kernel is via pip:

.. code-block:: bash

   pip install tf-kernel

Install from Source
-------------------

1. Clone the repository:

.. code-block:: bash

   git clone https://github.com/YOUR_ORG/tf-kernel.git
   cd tf-kernel

2. Install build dependencies:

.. code-block:: bash

   pip install scikit-build-core ninja

3. Build and install:

.. code-block:: bash

   make build-auto

Build Options
-------------

Target SM Architecture
^^^^^^^^^^^^^^^^^^^^^^

You can target specific GPU architectures to reduce build time:

.. code-block:: bash

   make build-sm80   # Ampere (A100, RTX 3090)
   make build-sm90   # Hopper (H100)
   make build-sm100  # Blackwell (RTX 5090)

Or use auto-detection:

.. code-block:: bash

   make build-auto

Limit Build Resources
^^^^^^^^^^^^^^^^^^^^^

To limit CPU and memory usage during build:

.. code-block:: bash

   make build MAX_JOBS=2 CMAKE_ARGS="-DTF_KERNEL_COMPILE_THREADS=1"

Verify Installation
-------------------

After installation, verify that tf-kernel is working:

.. code-block:: python

   import torch
   import tf_kernel

   # Check available operations
   print(dir(tf_kernel))

Troubleshooting
---------------

Segmentation fault with CUDA 12.6
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Update ptxas to 12.8. See `FlashAttention Issue #1453 <https://github.com/Dao-AILab/flash-attention/issues/1453>`_.

CUDA Runtime Not Found
^^^^^^^^^^^^^^^^^^^^^^

Set ``CUDA_HOME`` or ``CUDA_PATH`` environment variable, or ensure CUDA libraries are in ``LD_LIBRARY_PATH``.
