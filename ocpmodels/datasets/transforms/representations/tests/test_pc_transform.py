import pytest

import torch

from ocpmodels.datasets.materials_project import (
    materialsproject_devset,
    MaterialsProjectDataset,
)
from ocpmodels.datasets.lips import lips_devset, LiPSDataset
from ocpmodels.datasets.transforms import PointCloudToGraphTransform
from ocpmodels.common import package_registry


if package_registry["dgl"]:
    import dgl

    @pytest.fixture()
    def pc_data():
        data = {
            "node_feats": torch.rand(10, 5),
            "edge_feats": torch.rand(15, 2),
            "atomic_numbers": torch.randint(1, 100, (10,)),
            "coords": torch.rand(10, 3),
            "dataset": "FakeDataset",
        }
        return data

    @pytest.mark.dependency()
    def test_transform_init():
        t = PointCloudToGraphTransform("dgl")

    @pytest.mark.dependency(depends=["test_transform_init"])
    def test_dgl_create(pc_data):
        t = PointCloudToGraphTransform("dgl")
        data = t(pc_data)
        assert all([key in data for key in ["graph", "dataset"]])

    @pytest.mark.dependency(depends=["test_dgl_create"])
    def test_dgl_data_copy(pc_data):
        t = PointCloudToGraphTransform(
            "dgl",
            node_keys=["pos", "atomic_numbers", "node_feats"],
        )
        data = t(pc_data)
        graph = data.get("graph")
        assert all([key in data for key in ["graph", "dataset"]])
        assert all(
            [key in graph.ndata for key in ["pos", "atomic_numbers", "node_feats"]]
        )

    @pytest.mark.dependency(depends=["test_transform_init"])
    def test_dgl_transform_fail(pc_data):
        t = PointCloudToGraphTransform("dgl")
        del pc_data["coords"]
        with pytest.raises(AssertionError):
            t(pc_data)

    @pytest.mark.dependency(depends=["test_transform_init", "test_dgl_create"])
    def test_dgl_materials_project():
        dset = MaterialsProjectDataset(
            materialsproject_devset, transforms=[PointCloudToGraphTransform("dgl")]
        )
        sample = dset.__getitem__(0)
        assert "graph" in sample.keys()
        g = sample.get("graph")
        assert all([key in g.ndata for key in ["pos", "atomic_numbers"]])

    @pytest.mark.dependency(depends=["test_transform_init", "test_dgl_create"])
    def test_dgl_lips():
        dset = LiPSDataset(lips_devset, transforms=[PointCloudToGraphTransform("dgl")])
        sample = dset.__getitem__(0)
        assert "graph" in sample.keys()
        g = sample.get("graph")
        assert all([key in g.ndata for key in ["pos", "atomic_numbers", "force"]])
