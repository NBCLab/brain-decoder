Installation
============

Install the package from PyPI:

.. code-block:: bash

   pip install braindec

Install the plotting extras when you want to run the tutorials or make brain
surface figures:

.. code-block:: bash

   pip install "braindec[plotting]"

Development Install
-------------------

For local development, clone the repository and install it in editable mode:

.. code-block:: bash

   git clone https://github.com/jdkent/brain-decoder.git
   cd brain-decoder
   pip install -e ".[doc,plotting,test]"

Documentation Build
-------------------

Build the documentation locally without executing the examples:

.. code-block:: bash

   python -m sphinx -b html docs docs/_build/html

The gallery examples are not executed by default because the full NiCLIP
tutorial downloads model assets and may need GPU resources. To execute the
examples locally, set ``BRAINDEC_BUILD_GALLERY=1`` before building.
