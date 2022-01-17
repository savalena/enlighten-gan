def init_weights(module):
    module.weight.data.normal_(0.0, 0.02)