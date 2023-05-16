from model.FBSNet import FBSNet

def build_model(model_name, num_classes):
    if model_name == 'FBSNet':
        return Net(classes=num_classes)