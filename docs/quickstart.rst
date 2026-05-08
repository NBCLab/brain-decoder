Quickstart
==========

The main prediction API decodes a brain activation map into Cognitive Atlas
task labels. The example below shows the shape of a typical workflow.

.. code-block:: python

   import nibabel as nib
   import numpy as np

   from braindec.fetcher import download_bundle, get_data_dir
   from braindec.predict import image_to_labels

   work_dir = get_data_dir()
   download_bundle("example_prediction", destination_root=work_dir)

   activation_img = nib.load("path/to/activation_map.nii.gz")
   vocabulary = ["motor fMRI task paradigm", "language processing fMRI task paradigm"]
   vocabulary_emb = np.load("path/to/vocabulary_embeddings.npy")
   vocabulary_prior = np.full(len(vocabulary), 1.0 / len(vocabulary))

   predictions = image_to_labels(
       activation_img,
       model_path="path/to/model.pth",
       vocabulary=vocabulary,
       vocabulary_emb=vocabulary_emb,
       prior_probability=vocabulary_prior,
       topk=10,
       logit_scale=20.0,
   )

   print(predictions)

For an end-to-end workflow with the packaged example assets, HCP contrast maps,
hierarchical decoding, ROI characterization, custom vocabularies, and latent
space plots, see :doc:`auto_examples/02_niclip_demo`.
