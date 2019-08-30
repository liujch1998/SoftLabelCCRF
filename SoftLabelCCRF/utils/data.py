import pickle

def load_tokens (split, n_instances):
    
    with open('split/%s.tokens' % (split), 'r') as f:
        tokens = f.read().strip().split('\n')
    return tokens[:n_instances]

def load_instance (token):

    with open('cache/' + token, 'rb') as f:
        instance = pickle.load(f, encoding='latin1')
    return instance

