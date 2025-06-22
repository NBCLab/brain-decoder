# NiCLIP

`Warning:` README is currently in progress. Please check back later.

[[Paper]](https://doi.org/10.1101/2025.06.14.659706)

We present NiCLIP, a neuroimaging contrastive language-image pretraining model designed for predicting text from brain images. NiCLIP is built upon the NeuroConText model’s CLIP architecture while enhancing the text encoding with the assistance of BrainGPT, a pre-trained LLM fine-tuned on neuroscience papers.

## Approach

![NiCLIP](NiCLIP.png)

**Overview of the framework for training the text-to-brain model and decoding brain activation maps.**
(A) The text-to-brain CLIP model is trained using text and brain activation coordinates sourced from a collection of fMRI articles downloaded from PubMed Central. Pubget is used to download and preprocess the articles in a standardized format. Text embeddings are determined using a pre-trained LLM. In contrast, image embeddings is obtained by first calculating an MKDA-modeled activation brain map, and second applying a continuous brain parcellation defined by the DiFuMo 512 atlas. (B) The brain decoding model relies on a cognitive ontology to predict text from input brain activation. The embeddings of task names along with their definitions are extracted using an LLM transformer, while image features are determined using the DiFuMo brain parcellation. The text and image embeddings are processed through the pre-trained text and image encoders from CLIP, yielding new embeddings in a shared latent space. The posterior probability representing the predicted task from the input activation is calculated based on the similarity between the text and image embeddings in the shared latent space.

## Usage

In order to use the code, you will need to install all of the Python libraries
that are required. The required library and associated versions are available in `requirements.txt`.

The easiest way to install the requirements is with Conda.

```bash
# Install conda if you don't have it already
cd /path/to/brain-decoder

# Create a new conda environment and install the requirements
conda create -p /path/to/niclip_env pip python=3.12
conda activate /path/to/niclip_env
pip install -e .[all]
```

### Model training

To train a new model you need text and image embeddings.

### Use pre-trained model

### Predictions

## Citation

If you use this code in your research, please acknowledge this work by citing the
paper: https://doi.org/10.1101/2025.06.14.659706.

## Note

The CLIP model relies on the [NeuroConText](https://github.com/ghayem/NeuroConText) code, which is based on the [CLIP](https://github.com/openai/CLIP) code.
