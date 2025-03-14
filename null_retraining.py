# Standard library imports
import os
import random
import pickle

# Third-party imports
import numpy as np
# import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms
import torch_pruning as tp
from torch.utils.data.sampler import SubsetRandomSampler

import torch.utils
import torch.utils.data

# Local imports
import utils

def main(seed: int):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = torch.load('./saved_models/cifar100_vgg19_mag_based_9.pth')

    data_pth = '../data'
    #seed =  FIRST ARG 
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
    # analyze initial metrics: accuracy
    acc, loss = utils.eval(model=model, test_loader=test_loader, device=device)
    super_acc = utils.eval_superclass(model=model, superclass_labels=superclass_labels, test_loader=test_loader, device=device)

    # analyze initial metrics: entropy
    most_uncertain, entropies = utils.analyze_test_set_uncertainty(model, test_loader, device)
    entropies_avg = np.mean(np.array(entropies))
    print(entropies_avg)

    # analyze initial metrics: ece 
    binned_acc, binned_confidence, counts, ece_init = utils.compute_calibration_metrics(model, test_loader)

    accuracy = [acc]
    super_accuracy = [super_acc]
    avg_entropies = [entropies_avg]
    ece = [ece_init]


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

        most_uncertain, entrop = utils.analyze_test_set_uncertainty(model, test_loader, device, top_k=10)
        entrop = np.array(entrop)
        avg_entrop = np.mean(entrop)
        avg_entropies.append(avg_entrop)

        acc_in_bins, confidence_in_bins, counts, ece_decay = utils.compute_calibration_metrics(model, test_loader)
        ece.append(ece_decay)

        # only save one seed of model to do geometric comparisions 
        if seed == 42:
            torch.save(model.state_dict(), f'saved_models/injured_models/null_retraining/3000/3000_damaged_{i}.pth')
        ######################################################################
        # normal small subset of unseen data
        ######################################################################
        np.random.shuffle(valid_index)
        subset_indices = valid_index[:3000]
        small_sampler = SubsetRandomSampler(subset_indices)
        small_valid_loader = torch.utils.data.DataLoader(train_dst, sampler=small_sampler, batch_size=batch_size, num_workers=0)

        #####################################################################
        utils.train_model(model, small_valid_loader, test_loader, num_epochs, test_noise_dict=test_noise_dict, w_bkup=w_bkup, lr=0.01) 
        
        acc2, val_loss2 = utils.eval(model, test_loader, device=device)
        super_acc2 = utils.eval_superclass(model, superclass_labels, test_loader, device=device)
        accuracy.append(acc2)
        super_accuracy.append(super_acc2)

        most_uncertain_retrained, entrop_retrained = utils.analyze_test_set_uncertainty(model, test_loader, device, top_k=10)
        entrop_retrained = np.array(entrop_retrained)
        avg_entrop_retrained = np.mean(entrop_retrained)
        avg_entropies.append(avg_entrop_retrained)

        acc_in_bins, confidence_in_bins, counts, ece_retrained = utils.compute_calibration_metrics(model, test_loader)
        ece.append(ece_retrained)
        
        if seed == 42:
            torch.save(model.state_dict(), f'saved_models/injured_models/null_retraining/3000/3000_retrained_{i}.pth')
        

    results = {
        'accuracy': accuracy,
        'superclass_accuracy': super_accuracy,
        'entropy': avg_entropies,
        'calibration': ece
    }

    utils.save_array_results(results, seed)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True, help='Random seed')
    args = parser.parse_args()
    
    main(args.seed)
