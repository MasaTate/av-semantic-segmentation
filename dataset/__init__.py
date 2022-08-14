from .sound_cityscapes import SoundCityscapes
from .sound_cityscapes_auth import SoundCityscapesAuth
from .sound_cityscapes_identical_random import SoundCityscapesIdenticalRandom
from .sound_cityscapes_identical_fixed import SoundCityscapesIdenticalFixed
from .sound_cityscapes_different_fixed_rotate import SoundCityscapesDifferentFixedRotate
from .sound_cityscapes_single_crop import SoundCityscapesSingleCrop

def get_dataset(type, root, split, transform, sound_track, check_track=3, rotate=None, crop=True):
    if type == "SoundCityscapes":
        dataset = SoundCityscapes(root, split=split, transform=transform, sound_track=sound_track)
    elif type == "SoundCityscapesAuth":
        dataset = SoundCityscapesAuth(root, split=split, transform=transform, sound_track=sound_track)
    elif type == "IdenticalRandom":
        dataset = SoundCityscapesIdenticalRandom(root, split=split, transform=transform, sound_track=sound_track, check_track=check_track)
    elif type == "IdenticalFixed":
        dataset = SoundCityscapesIdenticalFixed(root, split=split, transform=transform, sound_track=sound_track, check_track=check_track)
    elif type == "DifferentFixedRotate":
        dataset = SoundCityscapesDifferentFixedRotate(root, split=split, transform=transform, sound_track=sound_track, check_track=check_track, rotate=rotate)
    elif type == "SingleCrop":
        dataset = SoundCityscapesSingleCrop(root, split=split, transform=transform, sound_track=sound_track, check_track=check_track, rotate=rotate, crop=crop)

    else:
        raise Exception("No such dataset")

    return dataset