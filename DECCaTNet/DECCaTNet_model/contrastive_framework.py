"""Following SimCLR CL structure
"""
import os
import pickle
import time

import torch
import torch.nn as nn
import torch.nn.functional as fn
from tqdm import tqdm
import pickle as pkl
import torchplot as plt

from DECCaTNet_model import DECCaTNet_model as DECCaTNet
from DECCaTNet_model.custom_dataset import PathDataset

"""
SeqCLR contrastive pre-training algortihm summary
Randomly transform a mininbatch of N channels into 2N augmented channels (two augmentations for each channel, creating pairs)

Channel Encoder: learned representations used for downstream tasks
- transform input into latent space representations, in this case it has the same length as the input, but four channels.
    this property is important due to it allowing several different length downstream tasks. 
- two different encoder architectures

Projector: Collapses output of encoder into a 32-dimensional point, this output is where the agreement is maximized

Contrastive loss: (normalized temperature-scaled cross entropy)
Have two outputs from the projector: z_1 and z_2 after sending in augmented views of x. 

In ssl_baselines_zac there is implemented on training algorithm for each model. We will also start off by doing this,
    maybe we will need to switch later

There are limited contrastive modules in the framework, but there is one example, with the contrastive adverserial loss


Overall solution from SSL_baselines_biosignals:

The model is called SeqCLR, however the overall implementation with pre-training etc is called SQnet, so it is a little
    bit tricky to follow the flow in the implementation.
augmentations are applied in the dataset by the overwriting _get_items_ function. Here, also one datapoint/sample is
    selected each time, meaning that this is where one and one channel is selected.
The datset is then loaded into a pytroch dataloader, where a for loop over the dataloader gives data for the pre-training
Assume that all these things is in place, when writing the contrastive framework. Will try to docuemtn well.

Overall flow:
load_dataset -> SQ dataset -> SSL dataset -> pytorch datalaoder -> into for loop in training function for SQnet
SeqCLR modules -> SeqCLR overallclass -> SQNet for pretraing -> pretrained as SQNet

This flow it the reason for no SeqCLR dataset or dataloader etc.
"""


# start by creating contrastive loss function, first is inspired by (Section 2.3 of proceedings.mlr.press/v136/mohsenvand20a/mohsenvand20a.pdf)
# which is a temperature contrastive loss

class ContrastiveLoss(nn.Module):
    """
    ContrastiveLoss module, can be used as a normal nn.Module as it has a forward function. The forward function takes
        two sets as input and returns a loss. This implementation is inspired by SimCLR Chen et al(2020) and SeqCLR Moshenvand
        et al (2021).
    """

    def __init__(self, temperature):
        """
        Initiializes the Contrastive loss module
        :param temperature: Learnable temperature parameter
        """
        super().__init__()  # call super to access and overwrite nn.Module methods
        self.tau = temperature
        self.BATCH_DIM = 0  # the dimension in z.size which has the batch size
        self.cos_sim = nn.CosineSimilarity(0)  # use cosine similiarity as similairity measurement

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, device, method='matrix'):
        if method == 'matrix':
            return self.forward_matrix(x1, x2, device)
        return self.forward_nested(x1, x2)

    def forward_matrix(self, x1: torch.Tensor, x2: torch.Tensor, device):
        """
        Attempt at faster implementation by using torch built in function sfor matrixes multiplication
        :param x1, x2: Latent space representations of all samples in batch. Positive pairs will have the same index in
            the two sets. All other indexes are considered as negative pairs
        :return: A loss over the entire batch wich maximizes agreement between positive paris and minimize agreement with
            negative pairs
            Inspiration: https://theaisummer.com/simclr/
        """
        batch_size = x1.size(self.BATCH_DIM)  # get the batch size in input

        # normalize
        x1 = fn.normalize(x1, p=2, dim=1)
        x2 = fn.normalize(x2, p=2, dim=1)

        # concatenate tensors to one matrix to enable matrix operations
        x1_x2 = torch.cat([x1, x2], dim=0)  # now shape is [2xbatch, output_encoder_shape]

        # calculate cosine similarity of all pairs, also negative pairs
        similarity_matrix = fn.cosine_similarity(x1_x2.unsqueeze(1), x1_x2.unsqueeze(0), dim=2)

        # Now we need to extract the positive and negative pairs in the batch, utilize that positive pairs are
        # shifted from the main diagonal with batch_size, first get only positive pairs for nominator
        sim_x1x2 = torch.diag(similarity_matrix, batch_size)
        sim_x2x1 = torch.diag(similarity_matrix, batch_size)

        pos_pairs = torch.cat([sim_x1x2, sim_x2x1], dim=0)

        # calculate nominator
        nominator = torch.exp(pos_pairs / self.tau)

        # get all negative and positive paris for denominator, exclude pairs with same samples (k=i)
        mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float()
        mask = mask.to(device)
        # use mask to get all pairs
        pairs = mask * similarity_matrix

        # calculate denominator
        denominator = torch.sum(torch.exp(pairs / self.tau), dim=1)

        # calculate losses for all samples in batch
        batch_loss = -torch.log(nominator / denominator)

        # average losses over all samples
        average_loss = torch.sum(batch_loss) / (2 * batch_size)
        return average_loss

    def forward_nested(self, z1: torch.Tensor, z2: torch.Tensor):
        """
        Called whenever ContrastiveLoss class is called after initialization
        In original paper, only z2 is assumed to be augmented, we augment both input signals
        :param z1,z2: Latent space representations of pairwise positive pairs as numpy arrays, same indexes should
            be positive pairs
        :return: ContrastiveLoss maximizing agreement between pairs and minimizing agreement with
            negative pairs
        """
        contrastive_loss = 0.  # set loss
        batch_size = z1.size(self.BATCH_DIM)  # get batch size from input

        # normalize data in batch as is has increased performance, however may be double
        # TODO: check if double normalization
        z1 = fn.normalize(z1, dim=1)
        z2 = fn.normalize(z2, dim=1)

        # TODO: check z1 before and after view, write comment
        z1 = z1.view(batch_size, -1)
        z2 = z2.view(batch_size, -1)

        for i in range(batch_size):
            # iterate over the entire batch, calculate loss with regard to one and one sample

            # calculte loss contirbutions from set z1 on z2
            cosine_pair = self.cos_sim(z1[i, :], z2[i, :])  # compute cosine similarity for positive pair
            num_1 = torch.exp(cosine_pair / self.tau)  # get numerator for loss function
            denom_1 = 0.  # define denominator in loss function
            for k in range(batch_size):
                # iterate over all samples with regard to sample i, calculate contirbution to loss from each sample
                cosine_z2 = self.cos_sim(z1[i, :], z2[k, :])
                denom_1 += torch.exp(cosine_z2 / self.tau)  # update denominator, include i == k for because z1 != z2
                if k != i:
                    # avoid calculating loss with regard to itself
                    cosine_z1 = self.cos_sim(z1[i, :], z1[k, :])
                    denom_1 += torch.exp(cosine_z1 / self.tau)  # update denominator with negative pairs from own set
            contrastive_loss += -1 * torch.log(num_1 / denom_1)  # update loss

            # calculate loss contirbutions from set z2 on z1
            num_2 = num_1  # as CosineSimilarity in pytorch is a cumulative function
            denom_2 = 0.  # define new demnominator
            for k in range(batch_size):
                # switch z1 and z2
                cosine_z1 = self.cos_sim(z2[i, :], z1[k, :])
                denom_2 += torch.exp(cosine_z1 / self.tau)
                if k != i:
                    cosine_z2 = self.cos_sim(z2[i, :], z2[k, :])
                    denom_2 += torch.exp(cosine_z2 / self.tau)
            contrastive_loss += -1 * torch.log(num_2 / denom_2)
        # avreage out loss for both batches, minus 1 because cosine_similiarty with itself is not included
        return contrastive_loss / (batch_size * 2. * (2. * batch_size - 1.))


class ContrastiveLossGPT(nn.Module):
    """
    ContrastiveLoss module, can be used as a normal nn.Module as it has a forward function. The forward function takes
        two sets as input and returns a loss. This implementation is inspired by SimCLR Chen et al(2020) and SeqCLR Moshenvand
        et al (2021).
    """

    def __init__(self, temperature):
        """
        Initiializes the Contrastive loss module
        :param temperature: Learnable temperature parameter
        """
        super(ContrastiveLossGPT, self).__init__()
        self.tau = temperature
        self.BATCH_DIM = 0  # the dimension in z.size which has the batch size
        self.cos_sim = nn.CosineSimilarity(0)  # use cosine similarity as similarity measurement

    def forward(self, z1: torch.Tensor, z2: torch.Tensor):
        """
        Called whenever ContrastiveLoss class is called after initialization
        In original paper, only z2 is assumed to be augmented, we augment both input signals
        :param z1,z2: Latent space representations of pairwise positive pairs as numpy arrays, same indexes should
            be positive pairs
        :return: ContrastiveLoss maximizing agreement between pairs and minimizing agreement with
            negative pairs
        """
        batch_size = z1.size(self.BATCH_DIM)  # get batch size from input
        print(z1.shape)
        z1 = fn.normalize(z1, dim=1)
        z2 = fn.normalize(z2, dim=1)

        # compute cosine similarities
        similarities = self.cos_sim(z1, z2) / self.tau

        # create mask to exclude positive pairs from the denominator
        mask = torch.eye(batch_size, device=z1.device)

        # compute numerator and denominator for z1 -> z2 loss
        numerator_1 = torch.exp(similarities) * (1 - mask)
        denominator_1 = torch.exp(similarities) + torch.sum(torch.exp(self.cos_sim(z1, z1) / self.tau), dim=1) - 1

        # compute numerator and denominator for z2 -> z1 loss
        numerator_2 = torch.exp(similarities.t()) * (1 - mask)
        denominator_2 = torch.exp(similarities.t()) + torch.sum(torch.exp(self.cos_sim(z2, z2) / self.tau), dim=1) - 1

        # compute losses
        loss_1 = -torch.sum(torch.log(numerator_1 / denominator_1))
        loss_2 = -torch.sum(torch.log(numerator_2 / denominator_2))

        # average over batch
        loss = (loss_1 + loss_2) / (2.0 * batch_size)

        return loss


# Next up: contrastive training framework
def pre_train_model(all_params,global_params):
    """
    :param dataset: ContrastiveAugmentedDataset for pre_training
    :param batch_size: batch size for pre_training
    :param train_split: percentage of dataset size which is training
    :param save_freq: how often model is saved (in epohs)
    :param shuffle: wether dataset should be shuffled or not
    :param trained_model_path: string path for already trained model
    :param temperature: temperature parameter in contrastiveloss_function, learnable
    :param learning_rate:
    :param num_workers: number of workers
    :param weight_decay:
    :param max_epochs:
    :param batch_print_freq: how often batch progress is printed in one epoch
    :param save_dir_model: save directory for all models #
    :param model_file_name: file name for this trained model.
    :param model_params: parameters in a dict for model to be trained
    :param time_process: If the different processes should be timed and printed / boolean
    :return:
    """
    """
    Needs:
    destination to load model, save model
    Hyperparameters for: Training loop, maybe model if not initialised and data_loaders
    Dependes a little bit about the overall architecture

    Need to implement som sort of support for GPUs and remote access to server.

    Need to figure out a good flow: first thought:
        - apply augmentations: either in dataloader or here in the pre_training
        - exstract two channels, to learn the dependencies between different channels. How should they be exstracted
            * from the same recodring at the same time? (will sucsefully learn the dependencies between channels at
                same time from same subject, maybe the most valuable of the proposed)
            * from the same subject, but at different times (will not focus on dependencies across time, but will learn to map
                samples from same subject closer in latent space. The dependencies learn will be time-invariant, which might not
                be something we are looking for. However frequencies-realated approaches have gotten a lot of success.
            * from different subjects at different times. Honestly dont know what will be learnt here. Will be taking two
                completely different recordings, apply transformations and try to learn to differentiate the latent space
                representations from two random channels from other random channel combinations.
            Need to rethink what are we trying to learn, what do we want to create a diversified latent space from??
        - send x (signals, recordings) through encoder
        - send x through projector which projects to a latent space.
        - us contrastive loss function on latent space created by projector.


    :return: None
    """
    params = all_params['pre_training']

    with open(params['ids_path'], 'rb') as fid:
        pre_train_ids = pickle.load(fid)
    dataset = PathDataset(path=params['ds_path'], ids_to_load=pre_train_ids, all_params=all_params, global_params=global_params)

    batch_size = params['batch_size']
    train_split = params['train_split']
    shuffle = params['SHUFFLE']
    trained_model_path = params['pretrained_model_path']
    save_freq = params['save_freq']
    temperature = params['temperature']
    learning_rate = params['learning_rate']
    weight_decay = params['weight_decay']
    num_workers = global_params['n_jobs']
    max_epochs = params['max_epochs']
    batch_print_freq = params['batch_print_freq']
    save_dir_model = params['save_dir_model']
    model_file_name = params['model_file_name']
    time_process = params['TIME_PROCESS']

    n_channels = global_params['n_channels']

    if not os.path.exists(save_dir_model):
        os.makedirs(save_dir_model)

    # load dataset
    train_set, val_set = dataset.get_splits(train_split)

    # create data_loaders, here batch size is decided
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle,
                                               num_workers=num_workers)
    # maybe alos num_workers)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    # check if cuda setup allowed:
    # device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # init model and check if weights already given
    model = DECCaTNet.DECCaTNet(all_params,global_params)
    if trained_model_path is not None:
        print(f'loading pre_trained model from {trained_model_path}')
        model.__init__from_dict(torch.load(trained_model_path))  # loaded already trained-model

    if torch.cuda.is_available():
        model.cuda()
        print("Moved model to CUDA")

    # get loss function and optimizer
    loss_func = ContrastiveLoss(temperature=temperature)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)  # TODO: check out betas for Adam and if Adam is the best choice

    # track losses
    losses = []
    time_names = ['batch', 'to_device', 'encoding', 'loss_calculation', 'backward', 'loss_update', 'delete', 'total']
    # iterative traning loop
    for epoch in range(max_epochs):
        model.train() # tells Pytorch Backend that model is trained (for example set dropout and have correct batchNorm)
        print('epoch number: ', epoch+1, 'of: ', max_epochs)
        epoch_loss = 0
        counter = 0  # counter for batch print.
        # start traning by looping through batches
        start_time = time.thread_time()
        for aug_1, aug_2, sample in tqdm(train_loader, position=0, leave=True):
            batch_time = time.thread_time()
            # transfer to GPU or CUDA
            x1, x2 = aug_1.to(device), aug_2.to(device)
            to_device_time = time.thread_time()
            # zero out existing gradients
            optimizer.zero_grad()
            # send through model and projector, asssume not splitted for now
            x1_encoded, x2_encoded = model(x1), model(x2)
            encoding_time = time.thread_time()
            # get loss, update weights and perform step
            loss = loss_func(x1_encoded, x2_encoded, device=device, method='matrix')
            loss_calculation_time = time.thread_time()
            loss.backward()
            backward_time = time.thread_time()
            optimizer.step()
            loss_update_time = time.thread_time()
            epoch_loss += loss.item()
            # free cuda memory
            del x1
            del x2
            del x2_encoded
            del x1_encoded
            del loss
            # torch.cuda.empty_cache()
            delete_time = time.thread_time()
            if time_process:
                if counter == 0:
                    time_values = [batch_time - start_time, to_device_time - batch_time, encoding_time - to_device_time,
                                   loss_calculation_time - encoding_time, backward_time - loss_calculation_time,
                                   loss_update_time - backward_time,
                                   delete_time - loss_update_time, delete_time - start_time]
                else:
                    time_values = [x + y for x, y in zip(time_values,
                                                         [batch_time - start_time, to_device_time - batch_time,
                                                          encoding_time - to_device_time,
                                                          loss_calculation_time - encoding_time,
                                                          backward_time - loss_calculation_time,
                                                          loss_update_time - backward_time,
                                                          delete_time - loss_update_time,
                                                          delete_time - start_time
                                                          ])]
                # check counter and print one some codition
                if counter % batch_print_freq == 0:
                    print('\n')
                    for x, y in zip(time_names, time_values):
                        average = y / ((counter + 1) * batch_size)
                        print(f'Average time used on {x} :  {average:.7f}')
            counter += 1
            start_time = time.thread_time()
        # TODO: decide how we can implement a validation_set for a SSL pretext task, SSL for biosignals has a porposal, not implemented
        # maybe validation test, early stopping or something similar here. Or some other way for storing model here.
        # for now we will use save_frequencie
        if epoch % save_freq == 0 and epoch != 0:
            print('epoch number: ', epoch+1, 'saving model  ')
            temp_save_path_model = os.path.join(save_dir_model, "temp_" + str(epoch+1) + "_" + model_file_name)
            torch.save(model.state_dict(), temp_save_path_model)
            # here is the solution, have the model as several modules; then different modules can be saved seperately
            temp_save_path_encoder = os.path.join(save_dir_model, "temp_encoder" + str(epoch+1) + "_" + model_file_name)
            torch.save(model.encoder.state_dict(), temp_save_path_encoder)
        epoch_loss = epoch_loss/counter # get average loss
        losses.append(epoch_loss)
    # save function for final model
    save_path_model = os.path.join(save_dir_model, model_file_name)
    torch.save(model.state_dict(), save_path_model)
    # here is the solution, have the model as several modules; then different modules can be saved seperately
    save_path_enocder = os.path.join(save_dir_model, "encoder_" + model_file_name)
    torch.save(model.encoder.state_dict(), save_path_enocder)

    # save all of parameters to pickelfile
    pickle_name = 'meta_data_and_params_' + model_file_name + '.pkl'

    meta_data_path = os.path.join(save_dir_model, pickle_name)
    with open(meta_data_path, 'wb') as outfile:
        pkl.dump({
            "avg_train_losses": losses,
            #"avg_train_accs": avg_train_accs, #TODO: check out avg_train_accs
            "save_dir_for_model": save_dir_model,
            "model_file_name": model_file_name,
            "batch_size": batch_size,
            "shuffle": shuffle,  # "num_workers": num_workers,
            "max_epochs": max_epochs,
            "learning_rate": learning_rate,
            'temperature':temperature,
            #"beta_vals": beta_vals, # TODO: check out betavals
            "weight_decay": weight_decay,
            "save_freq": save_freq,
            'noise_probability': dataset.noise_probability,
            'model_params': model_params,
            "channels": n_channels, #TODO:check where number of channels need to be changed
            'dataset_names':dataset.dataset_names,
            "sfreq": dataset.sfreq,
            'train_split':train_split,
            'already_trained_model':trained_model_path,
            'num_workers':num_workers,


        }, outfile)

    print('Traning Done!!')


def plot_avgs(avg_train_losses, avg_train_accs, avg_val_accs, plot_series_name, save_path):
    fig1, ax1 = plt.subplots()
    ax1.plot(avg_train_losses)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Average Loss")
    ax1.set_title(plot_series_name + ": Average Training Losses")
    plt.legend()
    plt.draw()
    loss_plot_save_path = os.path.join(save_path, plot_series_name + "_loss_visualization.png")
    fig1.savefig(loss_plot_save_path)

    fig2, ax2 = plt.subplots()
    ax2.plot(avg_train_accs, label="training")
    ax2.plot(avg_val_accs, label="validation")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Average Accuracy")
    ax2.set_title(plot_series_name + ": Average Prediction Accuracy")
    plt.legend()
    plt.draw()
    accuracy_plot_save_path = os.path.join(save_path, plot_series_name + "_accuracy_visualization.png")
    fig2.savefig(accuracy_plot_save_path)
    pass
