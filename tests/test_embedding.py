import nibabel as nib
import numpy as np

from braindec.embedding import ImageEmbedding


def _reference_volume_embedding(atlas_img, image_img):
    maps_data = atlas_img.get_fdata(dtype=np.float32)
    maps_data = np.nan_to_num(maps_data, nan=0.0, posinf=0.0, neginf=0.0)

    image_data = image_img.get_fdata(dtype=np.float32)
    image_data = np.nan_to_num(image_data, nan=0.0, posinf=0.0, neginf=0.0)
    if image_data.ndim == 3:
        image_data = image_data[..., None]

    maps_gram = np.tensordot(
        maps_data,
        maps_data,
        axes=([0, 1, 2], [0, 1, 2]),
    ).astype(np.float32)
    maps_gram += np.eye(maps_gram.shape[0], dtype=np.float32) * 1e-6
    xty = np.tensordot(maps_data, image_data, axes=([0, 1, 2], [0, 1, 2])).astype(
        np.float32
    )
    embeddings = np.linalg.solve(maps_gram, xty).T
    if embeddings.ndim == 1:
        embeddings = embeddings[None, :]

    return embeddings


def test_volume_embedding_matches_dense_reference_formula():
    rng = np.random.default_rng(13)
    atlas_data = rng.normal(size=(2, 2, 2, 3)).astype(np.float32)
    atlas_data[0, 0, 0, :] = 0.0
    image_data = rng.normal(size=(2, 2, 2)).astype(np.float32)
    image_data[0, 0, 0] = 10_000.0

    affine = np.eye(4)
    atlas_img = nib.Nifti1Image(atlas_data, affine)
    image_img = nib.Nifti1Image(image_data, affine)

    embedder = ImageEmbedding.__new__(ImageEmbedding)
    embedder.atlas_maps = atlas_img
    embedder._maps_cache = {}

    expected = _reference_volume_embedding(atlas_img, image_img)
    actual = embedder._generate_volume_embedding(image_img)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
