# Welcome

Thank you for checking out our project. 

The Data directory contains the model and partially results. If you want to use it please follow this steps:

## Joining and Extracting Split Tar Files

### 1. Join the files:

Use the `cat` command to concatenate the split parts back into a single tarball:

```bash
cat Data.tar.gz.part-* > Data.tar.gz
```

### 2. Untar the file:

```bash
tar xzvf Data.tar.gz
```

Please contact me: victor.lomas at iimas.unam.mx



