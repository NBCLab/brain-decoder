import importlib.util
import os.path as op
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FETCHER_PATH = ROOT / "braindec" / "fetcher.py"


def _load_fetcher_module(monkeypatch, tmp_path):
    braindec_pkg = types.ModuleType("braindec")
    utils_mod = types.ModuleType("braindec.utils")
    utils_mod.get_data_dir = lambda data_dir=None: str(tmp_path if data_dir is None else Path(data_dir))

    monkeypatch.setitem(sys.modules, "braindec", braindec_pkg)
    monkeypatch.setitem(sys.modules, "braindec.utils", utils_mod)

    spec = importlib.util.spec_from_file_location("fetcher_under_test", FETCHER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_materialized_path_to_local_path(monkeypatch, tmp_path):
    fetcher = _load_fetcher_module(monkeypatch, tmp_path)
    destination = fetcher._materialized_path_to_local_path(
        "/results/pubmed/model.pth",
        tmp_path / "downloads",
    )
    assert destination == tmp_path / "downloads" / "results" / "pubmed" / "model.pth"


def test_download_bundle_dispatches_all_assets(monkeypatch, tmp_path):
    fetcher = _load_fetcher_module(monkeypatch, tmp_path)
    calls = []

    def fake_download_asset(name, destination_root=".", overwrite=False, node_id=None, timeout=60):
        calls.append((name, Path(destination_root), overwrite, node_id, timeout))
        return [Path(destination_root) / name]

    monkeypatch.setattr(fetcher, "download_asset", fake_download_asset)

    downloaded = fetcher.download_bundle("example_prediction", destination_root=tmp_path, overwrite=True)
    expected_assets = fetcher.OSF_BUNDLES["example_prediction"]
    assert [call[0] for call in calls] == expected_assets
    assert downloaded == [tmp_path / asset for asset in expected_assets]


def test_download_osf_folder_recurses_into_nested_folders(monkeypatch, tmp_path):
    fetcher = _load_fetcher_module(monkeypatch, tmp_path)

    def fake_get_folder_item(node_id, remote_path, provider=None, timeout=60):
        assert remote_path == "results/pubmed"
        return {"id": "root-folder", "materialized_path": "/results/pubmed/"}

    def fake_iter_children(node_id=None, folder_id=None, provider=None, timeout=60):
        if folder_id == "root-folder":
            return iter(
                [
                    {
                        "id": "nested-folder",
                        "attributes": {
                            "kind": "folder",
                            "name": "subdir",
                            "materialized_path": "/results/pubmed/subdir/",
                        },
                        "links": {},
                    },
                    {
                        "id": "file-a",
                        "attributes": {
                            "kind": "file",
                            "name": "model-a.pth",
                            "materialized_path": "/results/pubmed/model-a.pth",
                        },
                        "links": {"download": "https://example.test/model-a"},
                    },
                ]
            )
        if folder_id == "nested-folder":
            return iter(
                [
                    {
                        "id": "file-b",
                        "attributes": {
                            "kind": "file",
                            "name": "model-b.pth",
                            "materialized_path": "/results/pubmed/subdir/model-b.pth",
                        },
                        "links": {"download": "https://example.test/model-b"},
                    }
                ]
            )
        raise AssertionError(f"Unexpected folder_id {folder_id}")

    downloaded = []

    def fake_download_to_file(url, destination, overwrite=False, timeout=60, chunk_size=None):
        downloaded.append((url, Path(destination), overwrite))
        return Path(destination)

    monkeypatch.setattr(fetcher, "_get_folder_item", fake_get_folder_item)
    monkeypatch.setattr(fetcher, "_iter_children", fake_iter_children)
    monkeypatch.setattr(fetcher, "_download_to_file", fake_download_to_file)

    results = fetcher.download_osf_folder("results/pubmed", destination_root=tmp_path, overwrite=True)

    assert results == [
        tmp_path / "results" / "pubmed" / "model-a.pth",
        tmp_path / "results" / "pubmed" / "subdir" / "model-b.pth",
    ]
    assert downloaded == [
        ("https://example.test/model-a", tmp_path / "results" / "pubmed" / "model-a.pth", True),
        ("https://example.test/model-b", tmp_path / "results" / "pubmed" / "subdir" / "model-b.pth", True),
    ]
