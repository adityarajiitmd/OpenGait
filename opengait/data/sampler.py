import math
import random
import torch
import torch.distributed as dist
import torch.utils.data as tordata

# batch_sample[0] : no. of person id
# batch_sample[1] : no. of pickle files to be taken
class TripletSampler(tordata.sampler.Sampler):
    # Takes the dataset (dataset), batch size (batch_size), and a flag for shuffling (batch_shuffle) as input.
    def __init__(self, dataset, batch_size, batch_shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        if len(self.batch_size) != 2:
            raise ValueError(
                 # It expects a tuple of length 2 representing the number of positive pairs (batch_size[0]) and the number of negative samples per positive pair (batch_size[1]).
                "batch_size should be (P x K) not {}".format(batch_size))
        self.batch_shuffle = batch_shuffle
        # Gets the number of GPUs in the distributed setting 
        self.world_size = dist.get_world_size()

        # Validates that the total batch size (batch_size[0] * batch_size[1]) is divisible by the world size to ensure balanced distribution of samples across GPUs.
        if (self.batch_size[0]*self.batch_size[1]) % self.world_size != 0:
            raise ValueError("World size ({}) is not divisible by batch_size ({} x {})".format(
                self.world_size, batch_size[0], batch_size[1]))
        self.rank = dist.get_rank()

    def __iter__(self):
        while True:
            # Initializes an empty list sample_indices to store sample indices for the current batch.
            sample_indices = []
            # The function sync_random_sample_list (likely a custom function for distributed random sampling) is used to get a list of batch_size[0] unique positive class labels (pid_list).
            pid_list = sync_random_sample_list(
                self.dataset.label_set, self.batch_size[0])

            for pid in pid_list:
                indices = self.dataset.indices_dict[pid]
                indices = sync_random_sample_list(
                    indices, k=self.batch_size[1])
                sample_indices += indices
 # it will shuffle the batch
            if self.batch_shuffle:
                sample_indices = sync_random_sample_list(
                    sample_indices, len(sample_indices))

            # this is used to work for the whole batch size
            total_batch_size = self.batch_size[0] * self.batch_size[1]
            total_size = int(math.ceil(total_batch_size /
                                       self.world_size)) * self.world_size
            sample_indices += sample_indices[:(
                total_batch_size - len(sample_indices))]

            sample_indices = sample_indices[self.rank:total_size:self.world_size]
            yield sample_indices

    def __len__(self):
        return len(self.dataset)

# This Python function, likely used within the TripletSampler class, is designed for distributed random sampling in a deep learning training scenario.
def sync_random_sample_list(obj_list, k, common_choice=False):
    # obj_list: The list of objects to sample from (e.g., positive class labels in TripletSampler).
    # k: The number of elements to sample from the list.
     # A flag indicating whether all GPUs should choose the same random samples (potentially for ensuring consistency).
    
    if common_choice:
        # Uses random.choices (Python 3.8+) to sample k elements with replacement from obj_list. This ensures all GPUs get the same random subset.
        idx = random.choices(range(len(obj_list)), k=k) 
        idx = torch.tensor(idx)
    if len(obj_list) < k:
        idx = random.choices(range(len(obj_list)), k=k)
        idx = torch.tensor(idx)
    else:
        idx = torch.randperm(len(obj_list))[:k]
        # if gpu is available, move idx to suda indexes
    if torch.cuda.is_available():
        idx = idx.cuda()

    # Uses distributed communication (torch.distributed.broadcast) to broadcast the idx tensor from rank 0 (potentially the master process) to
    # all other ranks (GPUs) in the distributed training setup. This ensures all GPUs have the same sampling indices.
    torch.distributed.broadcast(idx, src=0)
    # Converts the broadcasted tensor (idx) back to a Python list (idx.tolist()).
    idx = idx.tolist()

    # Uses list comprehension to iterate through the sample indices (idx) and access the corresponding objects from the original list (obj_list).
# Returns a list containing the sampled objects.
    return [obj_list[i] for i in idx]


class InferenceSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

        self.size = len(dataset)
        indices = list(range(self.size))

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        if batch_size % world_size != 0:
            raise ValueError("World size ({}) is not divisible by batch_size ({})".format(
                world_size, batch_size))

        if batch_size != 1:
            complement_size = math.ceil(self.size / batch_size) * \
                batch_size
            indices += indices[:(complement_size - self.size)]
            self.size = complement_size

        batch_size_per_rank = int(self.batch_size / world_size)
        indx_batch_per_rank = []

        for i in range(int(self.size / batch_size_per_rank)):
            indx_batch_per_rank.append(
                indices[i*batch_size_per_rank:(i+1)*batch_size_per_rank])

        self.idx_batch_this_rank = indx_batch_per_rank[rank::world_size]

    def __iter__(self):
        yield from self.idx_batch_this_rank

    def __len__(self):
        return len(self.dataset)


class CommonSampler(tordata.sampler.Sampler):
    def __init__(self,dataset,batch_size,batch_shuffle):

        self.dataset = dataset
        self.size = len(dataset)
        self.batch_size = batch_size
        if isinstance(self.batch_size,int)==False:
            raise ValueError(
                "batch_size shoude be (B) not {}".format(batch_size))
        self.batch_shuffle = batch_shuffle
        
        self.world_size = dist.get_world_size()
        if self.batch_size % self.world_size !=0:
            raise ValueError("World size ({}) is not divisble by batch_size ({})".format(
                self.world_size, batch_size))
        self.rank = dist.get_rank() 
    
    def __iter__(self):
        while True:
            indices_list = list(range(self.size))
            sample_indices = sync_random_sample_list(
                    indices_list, self.batch_size, common_choice=True)
            total_batch_size =  self.batch_size
            total_size = int(math.ceil(total_batch_size /
                                       self.world_size)) * self.world_size
            sample_indices += sample_indices[:(
                total_batch_size - len(sample_indices))]
            sample_indices = sample_indices[self.rank:total_size:self.world_size]
            yield sample_indices

    def __len__(self):
        return len(self.dataset)

# **************** For GaitSSB ****************
# Fan, et al: Learning Gait Representation from Massive Unlabelled Walking Videos: A Benchmark, T-PAMI2023
import random
class BilateralSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, batch_size, batch_shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.batch_shuffle = batch_shuffle

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        self.dataset_length = len(self.dataset)
        self.total_indices = list(range(self.dataset_length))

    def __iter__(self):
        random.shuffle(self.total_indices)
        count = 0
        batch_size = self.batch_size[0] * self.batch_size[1]
        while True:
            if (count + 1) * batch_size >= self.dataset_length:
                count = 0
                random.shuffle(self.total_indices)

            sampled_indices = self.total_indices[count*batch_size:(count+1)*batch_size]
            sampled_indices = sync_random_sample_list(sampled_indices, len(sampled_indices))

            total_size = int(math.ceil(batch_size / self.world_size)) * self.world_size
            sampled_indices += sampled_indices[:(batch_size - len(sampled_indices))]

            sampled_indices = sampled_indices[self.rank:total_size:self.world_size]
            count += 1

            yield sampled_indices * 2

    def __len__(self):
        return len(self.dataset)
