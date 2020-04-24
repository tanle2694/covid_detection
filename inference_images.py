from __future__ import print_function
import argparse
import torch.backends.cudnn as cudnn
import torch
import numpy as np
import torch.nn as nn
from torchvision import datasets, models, transforms
from dataloaders.dataloader import CTImageLoaderTest
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(args):

    validation_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    validation_dataset = CTImageLoaderTest(link_label_file=args.validation_data, image_size=args.input_size,
                                       root_folder=args.root_folder, transforms=validation_transform)

    vali_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.vali_batch_size, shuffle=False,
                                              num_workers=args.workers, drop_last=False)

    # Init and load model
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 3)
    checkpoint = torch.load(args.resume)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    n_correct = 0
    n_total = 0
    f_write = open(args.predict_result, 'w')
    with torch.no_grad():
        for i, (data, target, links) in enumerate(vali_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, pred = torch.max(output, dim=1)
            for j in range(len(links)):
                f_write.write("{}\t{}\t{}\n".format(links[j], target[j].item(), pred[j].item()))
                if pred[j].item() == target[j].item():
                    n_correct += 1
                n_total += 1
    f_write.close()
    print("Acc: ", n_correct / n_total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation_data", default="data/vali.txt")
    parser.add_argument("--root_folder", default="data")
    parser.add_argument("--input_size", default=256)
    parser.add_argument("--vali_batch_size", default=32)
    parser.add_argument("--seed", default=1)
    parser.add_argument("--workers", default=4)
    parser.add_argument("--resume", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--predict_result", default="data/result.txt")
    args = parser.parse_args()
    main(args)