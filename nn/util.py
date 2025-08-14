from mytorch import tensor
import numpy as np

class PackedSequence:
    
    '''
    Encapsulates a list of tensors in a packed seequence form which can
    be input to RNN and GRU when working with variable length samples
    
    ATTENTION: The "argument batch_size" in this function should not be confused with the number of samples in the batch for which the PackedSequence is being constructed. PLEASE read the description carefully to avoid confusion. The choice of naming convention is to align it to what you will find in PyTorch. 

    Args:
        data (Tensor):( total number of timesteps (sum) across all samples in the batch, # features ) 
        sorted_indices (ndarray): (number of samples in the batch for which PackedSequence is being constructed,) - Contains indices in descending order based on number of timesteps in each sample
        batch_sizes (ndarray): (Max number of timesteps amongst all the sample in the batch,) - ith element of this ndarray represents no.of samples which have timesteps > i
    '''
    def __init__(self,data,sorted_indices,batch_sizes):
        
        # Packed Tensor
        self.data = data # Actual tensor data

        # Contains indices in descending order based on no.of timesteps in each sample
        self.sorted_indices = sorted_indices # Sorted Indices
        
        # batch_size[i] = no.of samples which have timesteps > i
        self.batch_sizes = batch_sizes # Batch sizes
    
    def __iter__(self):
        yield from [self.data,self.sorted_indices,self.batch_sizes]
    
    def __str__(self,):
        return 'PackedSequece(data=tensor({}),sorted_indices={},batch_sizes={})'.format(str(self.data),str(self.sorted_indices),str(self.batch_sizes))

def pack_sequence(sequence): 
    '''
    Constructs a packed sequence from an input sequence of tensors.
    By default assumes enforce_sorted ( compared to PyTorch ) is False
    i.e the length of tensors in the sequence need not be sorted (desc).

    Args:
        sequence (list of Tensor): ith tensor in the list is of shape (Ti,K) where Ti is the number of time steps in sample i and K is the # features
    Returns:
        PackedSequence: data attribute of the result is of shape ( total number of timesteps (sum) across all samples in the batch, # features )
    '''
    
    # TODO: INSTRUCTIONS
    # Find the sorted indices based on number of time steps in each sample
    # Extract slices from each sample and properly order them for the construction of the packed tensor. __getitem__ you defined for Tensor class will come in handy
    # Use the tensor.cat function to create a single tensor from the re-ordered segements
    # Finally construct the PackedSequence object
    # REMEMBER: All operations here should be able to construct a valid autograd graph.
    tensorIdx_dic = {id(sequence[i]):i for i in range(len(sequence))}
    sorted_seq = sorted(sequence, reverse=True, key=lambda t: t.shape[0])
    sorted_idx = [tensorIdx_dic[id(sample)] for sample in sorted_seq]
    
    batchSizes = []
    segments = []

    for timestep in range(len(sorted_seq[0])):
        batch_size = 0
        for sample in sorted_seq:
            if timestep < len(sample): # len(sample) == sample.shape[0]
                # [timestep:timestep+1] retains shape in place when sliced
                segments.append(sample[timestep:timestep+1])
                batch_size += 1
        batchSizes.append(batch_size)

    packedSeq = tensor.cat(segments)

    return PackedSequence(packedSeq, np.array(sorted_idx), np.array(batchSizes))

def unpack_sequence(ps):
    '''
    Given a PackedSequence, this unpacks this into the original list of tensors.
    
    NOTE: Attempt this only after you have completed pack_sequence and understand how it works.

    Args:
        ps (PackedSequence)
    Returns:
        list of Tensors
    '''
    
    # TODO: INSTRUCTIONS
    # This operation is just the reverse operation of pack_sequences
    # Use the ps.batch_size to determine number of time steps in each tensor of the original list (assuming the tensors were sorted in a descending fashion based on number of timesteps)
    # Construct these individual tensors using tensor.cat
    # Re-arrange this list of tensor based on ps.sorted_indices
    packedSeq, sorted_idx, batchSizes = ps.data, ps.sorted_indices, ps.batch_sizes

    # Creates place holder for each list of tensors
    unpackedSeq = [[] for i in range(len(sorted_idx))]

    # Creating 'timestep_idx': proper indexing of packedSeq from batchSizes
    timestep_idx = [0]
    for i in range(len(batchSizes)):
        timestep_idx.append(timestep_idx[i]+batchSizes[i])

    # Creating 't_samples': list of each batch (e.g. [0_samples,1_samples,..,t_samples])
    t_samples = []
    for i in range(len(timestep_idx)-1):
        t_samples.append(packedSeq[timestep_idx[i]:timestep_idx[i+1]])

    # Creating 'unpackedSeq': Unpacking packedSeq into a list of samples 
    # i.e. [each element: list([0_sample_i,1_sample_i,..,t_sample_i])
    largest_batch_size = batchSizes[0]
    for i in range(largest_batch_size):
        for t_sample in t_samples:
            if i < len(t_sample):
                unpackedSeq[i].append(t_sample[i:i+1])

    # Concatenates each list in 'unpackedSeq' to retrieve time sequence samples
    for i in range(len(unpackedSeq)):
        unpackedSeq[i] = tensor.cat(unpackedSeq[i])

    # Re-order into original order of samples in the batch of training data
    unsortedSeq = [[] for i in range(len(sorted_idx))]
    for i in range(len(sorted_idx)):
        unsortedSeq[sorted_idx[i]] = unpackedSeq[i]

    return unsortedSeq