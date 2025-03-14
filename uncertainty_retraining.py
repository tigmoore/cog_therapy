import torch
import torch_pruning as tp
import torchvision.transforms as transforms
import os
import torch.nn.functional as F
import random 
import numpy as np 
from torch.utils.data import DataLoader, Dataset, Subset
import matplotlib.pyplot as plt
import pickle
import utils
from torchvision.utils import make_grid
from tqdm import tqdm
from typing import Tuple, Optional
import matplotlib.gridspec as gridspec
import torch.utils
import torch.utils.data

def main(seed: int):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = torch.load('./saved_models/cifar100_vgg19_mag_based_9.pth')
    encoder_model = torch.load('./saved_models/cifar100_vgg19_mag_based_9.pth')
    data_pth = '../data'

    g = utils.set_seeds(seed)
    batch_size = 64


    train_loader, valid_loader, train_dst, valid_index = utils.get_train_valid_loader(data_dir=data_pth, batch_size=batch_size, augment=True, random_seed=seed, valid_size=0.1, 
                                        shuffle=True, generator=g)

    test_loader, test_dst = utils.get_test_loader(data_dir=data_pth, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    superclass_labels = np.array([4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 
                            3, 14, 9, 18, 7, 11, 3, 9, 7, 11, 
                            6, 11, 5, 10, 7, 6, 13, 15, 3, 15, 
                            0, 11, 1, 10, 12, 14, 16, 9, 11, 5, 
                            5, 19, 8, 8, 15, 13, 14, 17, 18, 10, 
                            16, 4, 17, 4, 2, 0, 17, 4, 18, 17, 
                            10, 3, 2, 12, 12, 16, 12, 1, 9, 19, 
                            2, 10, 0, 1, 16, 12, 9, 13, 15, 13, 
                            16, 19, 2, 4, 6, 19, 5, 5, 8, 19, 
                            18, 1, 2, 15, 6, 0, 17, 8, 14, 13])

    model = model.to(device)

    most_uncertain, entropies = utils.analyze_test_set_uncertainty(model, test_loader, device)
    entropies_avg = np.mean(np.array(entropies))
    entropies_avg


    def extract_features(model, dataloader, device='cuda'):
        model = model.to(device)
        model.eval()
        
        features_list = []
        imgs = []
        activation = {}
        
        # Hook for layer 34
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
        
        # Register hook with proper name
        handle = model.features[34].register_forward_hook(get_activation('layer34'))
        
        with torch.no_grad():
            for batch in tqdm(dataloader):
                images = batch[0].to(device)
                imgs.append(images)
                _ = model(images)
                features = activation['layer34'].flatten(start_dim=1)
                features = F.normalize(features, p=2, dim=1)
                features_list.append(features.cpu())
        
        # Remove the hook
        handle.remove()
        
        return torch.cat(features_list, dim=0), torch.cat(imgs, dim=0).cpu()

    encoded_valid_set, imgs = extract_features(model, valid_loader)



    acc, loss = utils.eval(model=model, test_loader=test_loader, device=device)
    super_acc = utils.eval_superclass(model=model, superclass_labels=superclass_labels, test_loader=test_loader, device=device)

    binned_acc, binned_confidence, counts, ece_init = utils.compute_calibration_metrics(model, test_loader)

    # initial metrics 
    accuracy, super_accuracy, avg_entropy, ece = [acc], [super_acc], [entropies_avg], [ece_init]


    w_bkup = {}
    noise_dict = {}
    with open('noise_dict_holdout.pkl', 'rb') as file:
        noise_dict = pickle.load(file)

    num_el = [x.numel() for x in noise_dict.values()]
    percentage = np.arange(.1, 1.1, .1)
    # percentage = [.2]
    decay = 0.5
    num_epochs = 10
    learning_rate = 0.01




    for i in range(len(percentage)):

        test_noise_dict = dict()
            
        for j, sub in enumerate(noise_dict):

            # slicing t0 'percentage' and reassigning

            test_noise_dict[sub] = noise_dict[sub][:round(num_el[j]*percentage[i])]
        
        # print([b.numel() for b in test_noise_dict.values()])
        
        for x, y in model.named_modules():

            if isinstance(y, torch.nn.Conv2d) or isinstance(y, torch.nn.Linear):

                for n, w in y.named_parameters():
                    if n.endswith('weight'):
                        w_bkup[x] = w.data.reshape(-1)[test_noise_dict[x]] 

        for lsw in w_bkup.values():
            lsw*=decay
        
        model.to(device)
        with torch.no_grad():
            for xx, yy in model.named_modules():
                if xx in test_noise_dict:
                    yy.weight.data.view(-1)[test_noise_dict[xx]] = w_bkup[xx].to(device)
        print(f'iteration: {i}')         
        model.eval()
        acc1, val_loss = utils.eval(model=model, test_loader=test_loader, device=device)
        super_acc1 = utils.eval_superclass(model, superclass_labels, test_loader, device=device)
        
        accuracy.append(acc1)
        super_accuracy.append(super_acc1)
        #####################################################################
        most_uncertain, entrop = utils.analyze_test_set_uncertainty(model, test_loader, device, top_k=10)
        entrop = np.array(entrop)
        avg_entrop = np.mean(entrop)
        avg_entropy.append(avg_entrop)
        #####################################################################
        acc_in_bins, confidence_in_bins, counts, ece_decay = utils.compute_calibration_metrics(model, test_loader)
        ece.append(ece_decay)

        # if seed == 42: 
            # torch.save(model.state_dict(), f'saved_models/injured_models/entropy_retraining/500/500_damaged_{i}.pth')
        uncertain_indices = []
        for k, (img_index, ent, _, true_class, pred_class) in enumerate(most_uncertain, 1):
            uncertain_indices.append(img_index)

        uncertain_subset = Subset(test_dst, uncertain_indices)
        uncertain_dl = DataLoader(uncertain_subset)
        encoded_outputs, uncertain_img = extract_features(encoder_model, uncertain_dl)

        # Find top similar images in the validation set 
        top_similar = []
        for output in encoded_outputs:
            
            X_selected = output.repeat(5000, 1)
            encoded_valid_set = encoded_valid_set.to(device)
            X_selected = X_selected.to(device)
            loss = 'mse'

            if loss == 'cosine_proximity':
                similarity = 1 - F.cosine_similarity(encoded_valid_set, X_selected, dim=1)
            elif loss == 'mse':
                similarity = F.mse_loss(encoded_valid_set, X_selected, reduction='none').sum(dim=1)
            else:
                print('Unknown loss, using MSE.')
                similarity = F.mse_loss(encoded_valid_set, X_selected, reduction='none').sum(dim=1)


            similarity_cpu = similarity.cpu()
            similarity_sorted = torch.argsort(similarity_cpu)

            top = similarity_sorted[:300]
            top_similar.extend(top)

        sampling_indices = [valid_index[id] for id in top_similar]
        training_subset = Subset(train_dst, sampling_indices)
        entropy_dataloader = DataLoader(training_subset, batch_size=batch_size, shuffle=True)

        ############ train on images close to uncertain images from test set ####################
        utils.train_model(model, entropy_dataloader, test_loader, num_epochs, test_noise_dict=test_noise_dict, w_bkup=w_bkup, lr=0.01) 
        # retrained accuracy
        acc2, val_loss2 = utils.eval(model, test_loader, device=device)
        super_acc2 = utils.eval_superclass(model, superclass_labels, test_loader, device=device)
        accuracy.append(acc2)
        super_accuracy.append(super_acc2)

        # retrained entropy
        most_uncertain_retrained, entrop_retrained = utils.analyze_test_set_uncertainty(model, test_loader, device, top_k=30)
        entrop_retrained = np.array(entrop_retrained)
        avg_entrop_retrained = np.mean(entrop_retrained)
        avg_entropy.append(avg_entrop_retrained)

        # retrained ece
        acc_in_bins, confidence_in_bins, counts, ece_retrained = utils.compute_calibration_metrics(model, test_loader)
        ece.append(ece_retrained)


        # save retrained model
        # if seed == 42: 
            # torch.save(model.state_dict(), f'saved_models/injured_models/entropy_retraining/500/500_retrained_{i}.pth')

        results = {
        'accuracy': accuracy,
        'superclass_accuracy': super_accuracy,
        'entropy': avg_entropy,
        'calibration': ece
    }

    utils.save_array_results(results, seed)
  
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True, help='Random seed')
    args = parser.parse_args()
    
    main(args.seed)
