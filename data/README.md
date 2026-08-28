# Data

Large datasets do not belong in this repository. The default local runtime is
`E:/techjam26-runtime`, configured through environment variables copied from
`.env.example`.

Expected manifest columns:

```text
image_path,label,dataset,official_split,generator,manipulation_family,
source_image_group,width,height,file_format,tamper_mask_path
```

Canonical provenance labels are:

```text
authentic
tampered
fully_aigc
```

SID numeric labels are converted as `0 -> authentic`, `1 -> fully_aigc`, and
`2 -> tampered`. Preserve official dataset splits and keep related source-image
families together.

