import torch
import torch.nn as nn
import torch.nn.functional as F


class BeamEntry:
    "information about one single beam at specific time-step"
    def __init__(self):
        self.prTotal = 0  # blank and non-blank
        self.prNonBlank = 0  # non-blank
        self.prBlank = 0  # blank
        self.prText = 1  # LM score
        self.lmApplied = False  # flag if LM was already applied to this beam
        self.labeling = ()  # beam-labeling


class BeamState:
    "information about the beams at specific time-step"
    def __init__(self):
        self.entries = {}

    def norm(self):
        "length-normalise LM score"
        for (k, _) in self.entries.items():
            labelingLen = len(self.entries[k].labeling)
            self.entries[k].prText = self.entries[k].prText ** (1.0 / (labelingLen if labelingLen else 1.0))

    def sort(self):
        "return beam-labelings, sorted by probability"
        beams = [v for (_, v) in self.entries.items()]
        sortedBeams = sorted(beams, reverse=True, key=lambda x: x.prTotal*x.prText)
        return [x.labeling for x in sortedBeams]


class CTCDecoder:
    def __init__(self, vocabs):
        self.vocabs = vocabs
        pass

    def decode(self, probs, beam_width=25):
        blankIdx = len(self.vocabs)
        windows, shape = probs.shape

        last = BeamState()
        labeling = ()
        last.entries[labeling] = BeamEntry()
        last.entries[labeling].prBlank = 1
        last.entries[labeling].prTotal = 1
        for t in range(windows):
            curr = BeamState()
            bestLabelings = last.sort()[0:beam_width]

            for labeling in bestLabelings:

                prNonBlank = 0
                if labeling:
                    prNonBlank = last.entries[]
