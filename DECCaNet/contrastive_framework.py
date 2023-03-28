"""Following SimCLR CL structure
"""
import torch.nn as nn
import torch.nn.functional as fn
import torch
import numpy as np

from tqdm import tqdm
import os
import DECCaNet.DECCaNet_model as DECCaNet
import time

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

    def forward2(self, z1: torch.Tensor, z2: torch.Tensor):
        """
        Called whenever ContrastiveLoss class is called after initialization
        In original paper, only z2 is assumed to be augmented, we augment both input signals
        :param z1,z2: Latent space representations of pairwise positive pairs as numpy arrays, same indexes should
            be positive pairs
        :return: ContrastiveLoss maximizing agreement between pairs and minimizing agreement with
            negative pairs
        """
        contrastive_loss = 0.  # set loss
        batch_size = z1.size(0)  # get batch size from input

        # normalize data in batch as is has increased performance, however may be double
        # TODO: check if double normalization
        z1 = fn.normalize(z1, dim=1)
        z2 = fn.normalize(z2, dim=1)

        # compute cosine similarity between all pairs in the batch
        similarities = torch.matmul(z1, z2.t())

        # construct mask to exclude comparisons between identical samples
        mask = torch.eye(batch_size, dtype=torch.bool)

        # compute numerator and denominator of the contrastive loss for each sample in the batch
        numerator = torch.exp(similarities / self.tau)
        denominator = torch.sum(torch.exp(similarities / self.tau), dim=1) - torch.exp(
            torch.masked_select(similarities, mask)).sum()

        # compute the contrastive loss as the mean of the logarithm of the ratio of the numerator and denominator
        contrastive_loss = -torch.mean(torch.log(torch.masked_select(numerator, ~mask) / denominator))

        return contrastive_loss

    def forward(self, z1: torch.Tensor, z2: torch.Tensor):
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
def pre_train_model(dataset, batch_size, train_split, save_freq, shuffle, trained_model_path, temperature,
                    learning_rate, weight_decay,
                    num_workers, max_epochs, batch_print_freq, save_dir_model, model_file_name, model_params, time_process):
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
    :param save_dir_model: save directory for all models # TODO: check if empty and create new if empty
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
    # TODO seams like output from dataloader is planned as x , [x_1_aug, x_2_aug], need confirmation on this.
    # TODO: get confirmation on where augmentations are applied

    # load dataset
    train_set, val_set = dataset.get_splits(train_split)

    # create data_loaders, here batch size is decided
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle,
                                               num_workers=num_workers)
    # maybe alos num_workers)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    # check if cuda setup allowed:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # init model and check if weights already given
    if trained_model_path is not None:
        model = test_model.__init__from_dict(torch.load(trained_model_path))  # loaded already trained-model
    else:
        model = DECCaNet.DECCaNet(batch_size=batch_size)

    if torch.cuda.is_available():
        model.cuda()

    # get loss function and optimizer
    loss_func = ContrastiveLoss(temperature=temperature)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)  # TODO: check out betas for Adam and if Adam is the best choice

    # track losses
    losses = []
    time_names = ['batch', 'to_device', 'encoding', 'loss_calculation', 'loss_update', 'delete']
    # iterative traning loop
    for epoch in tqdm(range(max_epochs)):
        print('epoch number: ', epoch, 'of: ', max_epochs)
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
            loss = loss_func(x1_encoded, x2_encoded)
            loss_calculation_time = time.thread_time()
            loss.backward()
            optimizer.step()
            loss_update_time = time.thread_time()
            # free cuda memory
            del x1
            del x2
            del x2_encoded
            del x1_encoded
            del loss
            torch.cuda.empty_cache()
            delete_time = time.thread_time()
            if time_process:
                if counter == 0:
                    time_values = [batch_time-start_time, to_device_time- batch_time, encoding_time - to_device_time,
                                   loss_calculation_time-encoding_time, loss_update_time - loss_calculation_time,
                                   delete_time-loss_update_time]
                else:
                    time_values = [x + y for x, y in zip(time_values,
                                                         [batch_time - start_time, to_device_time - batch_time,
                                                          encoding_time - to_device_time,
                                                          loss_calculation_time - encoding_time,
                                                          loss_update_time - loss_calculation_time,
                                                          delete_time - loss_update_time])]
                # check counter and print one some codition
                if counter % batch_print_freq == 0:
                    print('\n')
                    for x, y in zip(time_names, time_values):
                        average = y / ((counter + 1)*batch_size)
                        print('Average time used on', x, ':', round(average)
            counter += 1
            start_time = time.thread_time()
        # TODO: decide how we can implement a validation_set for a SSL pretext task, SSL for biosignals has a porposal, not implemented
        # maybe validation test, early stopping or something similar here. Or some other way for storing model here.
        # for now we will use save_frequencie
        if epoch % save_freq == 0 and epoch != 0:
            print('epoch number: ', epoch, 'saving model  ')
            temp_save_path_model = os.path.join(save_dir_model, "temp_" + str(epoch) + "_" + model_file_name)
            torch.save(model.state_dict(), temp_save_path_model)
            # here is the solution, have the model as several modules; then different modules can be saved seperately
            temp_save_path_encoder = os.path.join(save_dir_model, "temp_encoder" + str(epoch) + "_" + model_file_name)
            torch.save(model.encoder.state_dict(), temp_save_path_encoder)

        losses.append(loss)
    # save function for final model
    save_path_model = os.path.join(save_dir_model, model_file_name)
    torch.save(model.state_dict(), save_path_model)
    # here is the solution, have the model as several modules; then different modules can be saved seperately
    save_path_enocder = os.path.join(save_dir_model, "encoder_" + model_file_name)
    torch.save(model.encoder.state_dict(), save_path_enocder)

    # save all of parameters to pickelfile
    # Want to include
    save_path_model
    save_path_enocder
    save_dir_model
    losses
    eval_losses  # TODO
    batch_size, save_freq, shuffle, trained_model_path, temperature, learning_rate,
    weight_decay, max_epochs, batch_print_freq, save_dir_model, model_file_name

    # then biosignals write a lot of metadata to a pickel file, which might not be stupid # TODO: check this out

    print('Traning Done!!')
