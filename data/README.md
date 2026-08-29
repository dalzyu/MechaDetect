# Data

Large datasets do not belong in this repository. The default local runtime is
`E:/techjam26-runtime`, configured through environment variables copied from
`.env.example`.

Expected manifest columns:

```text
image_path,label,dataset,official_split,generator,manipulation_family,
source_image_group,width,height,file_format,tamper_mask_path
```

Canonical source-provenance labels are:

```text
authentic
tampered
fully_aigc
```

The Track 5 image-level target is binary:

```text
authentic -> 0
tampered  -> 1
fully_aigc -> 1
```

SID numeric labels are converted as `0 -> authentic`, `1 -> fully_aigc`, and
`2 -> tampered`. Preserve the subtype label in manifests, but train the image
head on the binary target and keep related source-image families together.

