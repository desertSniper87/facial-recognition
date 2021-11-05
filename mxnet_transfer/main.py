import insightface


from mxnet.gluon.model_zoo.vision import mobilenet1_0
pretrained_net = mobilenet1_0(pretrained=True)
print(pretrained_net)


pretrained_net = insightface.model_zoo.get_model('antelopev2', download=True)
print(pretrained_net)

