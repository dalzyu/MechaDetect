from enum import IntEnum


class Provenance(IntEnum):
    AUTHENTIC = 0
    TAMPERED = 1
    FULLY_AIGC = 2


class Transformation(IntEnum):
    JPEG = 0
    BLUR = 1
    RESIZE = 2
    NOISE = 3
    COLOR = 4
    CROP = 5
    POISSON = 6
    GRAIN = 7
    HALFTONE = 8
    ENTROPY = 9


PROVENANCE_NAMES = tuple(item.name.lower() for item in Provenance)

SID_LABEL_TO_PROVENANCE = {
    0: Provenance.AUTHENTIC,
    1: Provenance.FULLY_AIGC,
    2: Provenance.TAMPERED,
}

SEVERITY_VALUES = {
    Transformation.JPEG: (90.0, 70.0, 50.0, 30.0),
    Transformation.BLUR: (0.5, 1.0, 2.0),
    Transformation.RESIZE: (0.5, 0.25),
    Transformation.NOISE: (0.02, 0.05, 0.10),
    Transformation.POISSON: (0.02, 0.05, 0.10),
    Transformation.GRAIN: (0.025, 0.05, 0.09),
    Transformation.HALFTONE: (0.06, 0.12, 0.20),
    Transformation.ENTROPY: (0.02, 0.05, 0.10),
}
