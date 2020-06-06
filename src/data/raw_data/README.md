# Raw Data directory structure
Note that, since we perform synthesis on a single directory at a time,
the PyTorch ImageFolder expects a directory structure like the
following:
```
+ metal-root/
  |-+ metal/
```

Making these alterations to trashnet/data/dataset-resized, however,
would require additional modifications to the trashnet source code to
account for the new directory structure during trashnet
training/validation/test. As such, we opt for the suggested fix as
mentioned by: https://github.com/pytorch/vision/issues/669.
