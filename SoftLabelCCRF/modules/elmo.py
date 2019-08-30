import torch
import torch.nn as nn

from allennlp.modules.elmo import Elmo

class ElmoWrapper (nn.Module):

    def __init__ (self, args):

        super(ElmoWrapper, self).__init__()

        options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
        self.elmo = Elmo(options_file, weight_file, 2, dropout=0.0).to(args.device)  # 2 layers
        self.elmo.eval()

    def forward (self, tokenid):
        '''
        > tokenid (batch, seqlen, .) int
        < emb (batch, seqlen, d_lang)
        < mask (batch, seqlen) bool
        '''

        with torch.no_grad():
            emb = self.elmo(tokenid)['elmo_representations'][-1]
        mask = tokenid[:,:,0] != 0
        return emb, mask

