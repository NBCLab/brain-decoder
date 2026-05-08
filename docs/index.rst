braindec — NiCLIP documentation
================================

**braindec** is the Python package for
`NiCLIP <https://doi.org/10.1101/2025.06.14.659706>`_, a contrastive
language–image pre-training model that decodes brain activation maps into
cognitive task descriptions from the
`Cognitive Atlas <https://www.cognitiveatlas.org/>`_ ontology.

.. toctree::
   :maxdepth: 1
   :caption: User guide

   installation
   quickstart
   auto_examples/index

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api

Installation
------------

.. code-block:: bash

   pip install braindec[plotting]

Quickstart
----------

Download the example assets and run functional decoding in a few lines:

.. code-block:: python

   from braindec.fetcher import download_bundle, get_data_dir
   from braindec.predict import image_to_labels

   work_dir = get_data_dir()
   download_bundle("example_prediction", destination_root=work_dir)

   # … construct paths, load model, then:
   task_df = image_to_labels(
       my_activation_map,
       model_path=model_fn,
       vocabulary=vocabulary,
       vocabulary_emb=vocabulary_emb,
       prior_probability=vocabulary_prior,
       topk=10,
       logit_scale=20.0,
   )
   print(task_df)

See the :doc:`examples gallery <auto_examples/index>` for a full walkthrough.

Citation
--------

.. code-block:: text

   Peraza et al. (2025). NiCLIP: Neuroimaging contrastive language-image
   pretraining model for predicting text from brain activation images.
   bioRxiv. https://doi.org/10.1101/2025.06.14.659706

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
