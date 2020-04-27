from allennlp.common import Registrable


class CommonBeamSearch(Registrable):

    def __init__(self, beam_size: int = 10):
        self.beam_size = beam_size

    def config_beam(self, end_index: int, max_steps: int):
        raise NotImplemented
