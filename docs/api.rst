API Reference
=============

Core prediction
---------------

.. automodule:: braindec.predict
   :members: image_to_labels, image_to_labels_hierarchical, preprocess_image

Embeddings
----------

.. automodule:: braindec.embedding
   :members: ImageEmbedding, TextEmbedding

Cognitive Atlas
---------------

.. automodule:: braindec.cogatlas
   :members: CognitiveAtlas

Data fetching
-------------

.. automodule:: braindec.fetcher
   :members: download_bundle, download_asset, download_osf_file,
             download_osf_folder, get_available_assets,
             get_cogatlas_tasks, get_cogatlas_concepts

Model
-----

.. automodule:: braindec.model
   :members: CLIP, build_model

Utilities
---------

.. automodule:: braindec.utils
   :members: get_data_dir, images_have_same_fov
