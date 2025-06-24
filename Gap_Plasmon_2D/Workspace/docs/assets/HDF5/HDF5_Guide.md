---
author:
- Yann Chardon-Grossard
date: June 2025
title: Hierarchical Data Format Version 5 (HDF5)
---

# Key Concepts

> An HDF5 file is a container for two kinds of objects: *datasets*,
> which are collections of data in array form, and *groups*, which are
> folder‐like containers holding datasets and other groups. Groups
> behave like dictionaries, and datasets like NumPy arrays.

[Reference: h5py Quick
Start](https://docs.h5py.org/en/stable/quick.html).

# What is HDF5?

HDF5 (Hierarchical Data Format version 5) is a binary format optimized
for storing large volumes of numerical data, together with an API (via
`h5py` or `PyTables` in Python) that provides:

- *Hierarchical* organization into groups and datasets, similar to a
  file system;

- the ability to add *attributes* (metadata) to any object;

- high‐performance *random access* thanks to chunking and compression.

# 1. Hierarchical Model

- **Groups** (`Group`): folder‐like containers.

- **Datasets** (`Dataset`): multidimensional arrays (`numpy.ndarray`).

- **Attributes** (`attrs`): key–value pairs attached to a group or
  dataset.

Example structure:

    /                             % root
    ├─ measurements/              % group
    │  ├─ time             (dataset)
    │  └─ temperature      (dataset)
    ├─ parameters/                % group
    │  ├─ voltage          (dataset)
    │  └─ current          (dataset)
    └─ metadata                   % group
       ├─ experiment_date (attribute)
       └─ operator        (attribute)

# 2. Why HDF5 Excels at Organizing Data

HDF5 delivers clear and efficient data management thanks to several
strengths:

1.  **Natural Organization**  
    You arrange your data in a simple tree structure, like folders and
    files on your computer. For example, `/measurements/temperature` or
    `/parameters/pressure`. This lets you locate and navigate directly
    to the information you need.

2.  **Built‐in Metadata**  
    Each group or dataset can carry its own labels (attributes), such as
    measurement units, creation date, or a description. All relevant
    information stays together in one place, without external sidecar
    files.

3.  **Partial, High‐Performance Access**  
    When your arrays are very large, HDF5 can split them into “chunks”
    and compress them. You can read or write a single chunk without
    loading the entire dataset into memory. The result: fast access,
    even for multi‐gigabyte datasets.

4.  **Interoperability**  
    HDF5 is a standard format recognized by many tools and languages (C,
    Fortran, MATLAB, R, Julia, etc.). You can share your file and open
    it elsewhere without complex conversion.

5.  **Scalability**  
    Whether your data are a few megabytes or multiple terabytes, HDF5
    scales. Performance remains high for both data access and updates,
    thanks to its internal indexing and caching mechanisms.
