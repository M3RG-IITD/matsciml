# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License
from __future__ import annotations


from matsciml.datasets.ocp_datasets import IS2REDataset, S2EFDataset


def test_base_s2ef_read():
    """
    This test will try and obtain the first and last elements
    of the dataset, as well as check that you can get the length
    of the dataset.
    """
    # no transforms
    dset = S2EFDataset.from_devset()
    # get the first entry
    data = dset.__getitem__(0)
    assert all([key in data for key in ["targets", "target_types"]])
    assert all([key in data.keys() for key in ["pos", "force"]])


def test_base_is2re_read():
    """
    This test will try and obtain the first and last elements
    of the dev IS2RE dataset and check its length
    """
    dset = IS2REDataset.from_devset()
    # get the first entry
    data = dset.__getitem__(0)
    assert all([key in data for key in ["targets", "target_types"]])
    assert "pos" in data.keys()
    assert all([key in data["targets"] for key in ["energy_relaxed", "energy_init"]])


def test_is2re_collate():
    """
    This function tests for the ability for an IS2RE dataset
    to be properly batched.
    """
    dset = IS2REDataset.from_devset()
    unbatched = [dset.__getitem__(i) for i in range(5)]
    batched = dset.collate_fn(unbatched)
    # check there are 5 samples
    assert len(batched["natoms"]) == 5
    # check one of the label shapes is correct
    assert batched["targets"]["energy_init"].size(0) == 5


def test_s2ef_collate():
    """
    This function tests for the ability for an S2EF dataset
    to be properly batched.
    """
    dset = S2EFDataset.from_devset()
    unbatched = [dset.__getitem__(i) for i in range(5)]
    batched = dset.collate_fn(unbatched)
    # check there are 5 graphs
    assert len(batched["natoms"]) == 5
    # check one of the label shapes is correct
    assert batched["targets"]["energy"].size(0) == 5
    num_nodes = sum(batched["natoms"])
    assert batched["targets"]["force"].shape == (num_nodes, 3)


def test_s2ef_target_keys():
    dset = S2EFDataset.from_devset()
    assert dset.target_keys == {"regression": ["energy", "force"]}


def test_is2re_target_keys():
    dset = IS2REDataset.from_devset()
    assert dset.target_keys == {"regression": ["energy_init", "energy_relaxed"]}
