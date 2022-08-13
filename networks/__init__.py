from . import model

def get_model(type, num_class, in_channel):
    if type == "audioToSeman":
        net = model.audioToSeman(num_class, in_channel)
    elif type == "audioToSemanSep":
        net = model.audioToSemanSep(num_class, in_channel)
    elif type == "singleAudioToSeman":
        net = model.singleAudioToSeman(num_class, in_channel)
    else:
        raise Exception("No such model")

    return net