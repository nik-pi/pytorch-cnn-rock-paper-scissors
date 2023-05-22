import torch
import rps

if __name__ == '__main__':
    idx_to_class={0: 'paper', 1: 'rock', 2: 'scissors'}
    classes=['paper', 'rock', 'scissors']

    data_transforms = rps.data_transforms
    model = rps.load_model()

    with torch.inference_mode():
        rps.live_inference(model, data_transforms, classes)