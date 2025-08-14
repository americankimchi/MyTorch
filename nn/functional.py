import numpy as np

from mytorch import tensor
from mytorch.autograd_engine import Function

# Allow broadcasting: a_shape, b_shape are tuples
def check_broadcastable_shapes(a_shape, b_shape):
    if len(a_shape) >= len(b_shape):
        longer_shape = a_shape
        shorter_shape = b_shape
    else:
        longer_shape = b_shape
        shorter_shape = a_shape
    diff_dim = len(longer_shape)-len(shorter_shape)
    for dim in range(len(shorter_shape)-1, -1, -1):
        # Right align shapes and if their dimensions differ
        if shorter_shape[dim] != longer_shape[dim+diff_dim]:
            # Only dimension of 1 is broadcastable if they differ
            if shorter_shape[dim] != 1:
                return False
    return True

# # grad is a numpy object
# # shape = shape of original matrix (tuple)
# def unbroadcast(grad, shape):
#   """
#   내가 만든 unbroadcast. 밑에는 2 dimensional 이상 고려 안한거같음
#   """
#     gradShape = grad.shape
#     diff_dim = len(gradShape)-len(shape)

#     # Loop each dimension(axis) backwards
#     for dim in range(len(shape)-1, -1, -1):
#         if shape[dim] != grad.shape[dim+diff_dim]:
#             grad = np.sum(grad, axis=dim+diff_dim)
#         # e.g. (15,3,) -> (15,3,1) if shape = (3,1) instead of (3,)
#         if shape[dim] == 1:
#             # Change to list
#             new_shape = list(gradShape)
#             # because tuple cannot be modified
#             new_shape[dim+diff_dim] = 1
#             grad = grad.reshape(tuple(new_shape))
#     # Removing leftmost dimensions
#     for i in range(diff_dim):
#         # Sum along left most axis
#         grad = np.sum(grad, axis=0)
        
#     return grad

def unbroadcast(grad, shape, to_keep=0):
    while len(grad.shape) != len(shape):
        grad = grad.sum(axis=0)
    for i in range(len(shape) - to_keep):
        if grad.shape[i] != shape[i]:
            grad = grad.sum(axis=i, keepdims=True)
    return grad

class Transpose(Function):
    @staticmethod
    def forward(ctx, a):
        if not len(a.shape) == 2:
            raise Exception("Arg for Transpose must be 2D tensor: {}".format(a.shape))
        requires_grad = a.requires_grad
        b = tensor.Tensor(a.data.T, requires_grad=requires_grad,
                                    is_leaf=not requires_grad)
        return b

    @staticmethod
    def backward(ctx, grad_output):
        return tensor.Tensor(grad_output.data.T)

class Reshape(Function):
    @staticmethod
    def forward(ctx, a, shape):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Reshape must be tensor: {}".format(type(a).__name__))
        # New context manager variable defined here
        ctx.shape = a.shape
        requires_grad = a.requires_grad
        c = tensor.Tensor(a.data.reshape(shape), requires_grad=requires_grad,
                                                 is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        return tensor.Tensor(grad_output.data.reshape(ctx.shape))

class Log(Function):
    @staticmethod
    def forward(ctx, a):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Log must be tensor: {}".format(type(a).__name__))
        ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        c = tensor.Tensor(np.log(a.data), requires_grad=requires_grad,
                                          is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        return tensor.Tensor(grad_output.data / a.data)

class Flatten(Function):
    @staticmethod
    def forward(ctx, a):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Reshape must be tensor: {}".format(type(a).__name__))
        # New context manager variable defined here
        ctx.shape = a.shape
        flattened = a.data.reshape(a.shape[0], np.prod(a.shape[1:]))
        requires_grad = a.requires_grad
        c = tensor.Tensor(flattened, requires_grad=requires_grad,
                                     is_leaf=not requires_grad)

        return c

    @staticmethod
    def backward(ctx, grad_output):
        return tensor.Tensor(grad_output.data.reshape(ctx.shape))

"""EXAMPLE: This represents an Op:Add node to the comp graph.

See `Tensor.__add__()` and `autograd_engine.Function.apply()`
to understand how this class is used.

Inherits from:
    Function (autograd_engine.Function)
"""
class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that both args are tensors
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))

        # # Check that args have same shape
        # if check_broadcastable_shapes(a.data.shape, b.data.shape) == False:
        #     raise Exception("Both args must have broadcastable sizes: {}, {}".format(a.shape, b.shape))

        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)

        # Create addition output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data + b.data, requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve forward inputs that we stored
        a, b = ctx.saved_tensors

        # calculate gradient of output w.r.t. each input
        grad_a = np.ones(a.shape) * grad_output.data
        grad_b = np.ones(b.shape) * grad_output.data

        if a.data.shape != grad_a.shape:
            grad_a = unbroadcast(grad_a, a.data.shape)
        if b.data.shape != grad_b.shape:
            grad_b = unbroadcast(grad_b, b.data.shape)

        # the order of gradients returned should match the order of the arguments
        return tensor.Tensor(grad_a), tensor.Tensor(grad_b)


class Sub(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that both args are tensors
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))

        # # Check that args have same shape
        # if check_broadcastable_shapes(a.data.shape, b.data.shape) == False:
        #     raise Exception("Both args must have broadcastable sizes: {}, {}".format(a.shape, b.shape))

        ctx.save_for_backward(a, b)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data - b.data, requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return c


    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = np.ones(a.shape) * grad_output.data
        grad_b = -(np.ones(b.shape) * grad_output.data)

        if a.data.shape != grad_a.shape:
            grad_a = unbroadcast(grad_a, a.data.shape)
        if b.data.shape != grad_b.shape:
            grad_b = unbroadcast(grad_b, b.data.shape)

        return tensor.Tensor(grad_a), tensor.Tensor(grad_b)

# TODO: Implement more Functions below
class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that both args are tensors
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))

        # # Check that args have same shape
        # if check_broadcastable_shapes(a.data.shape, b.data.shape) == False:
        #     raise Exception("Both args must have broadcastable sizes: {}, {}".format(a.shape, b.shape))

        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)

        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(np.multiply(a.data, b.data), requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve forward inputs that we stored
        a, b = ctx.saved_tensors

        # calculate gradient of output w.r.t. each input
        grad_a = np.multiply(grad_output.data, b.data)
        grad_b = np.multiply(grad_output.data, a.data)

        if a.data.shape != grad_a.shape:
            grad_a = unbroadcast(grad_a, a.data.shape)
        if b.data.shape != grad_b.shape:
            grad_b = unbroadcast(grad_b, b.data.shape)

        # the order of gradients returned should match the order of the arguments
        return tensor.Tensor(grad_a), tensor.Tensor(grad_b)

class Div(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that both args are tensors
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))

        # # Check that args have same shape
        # if check_broadcastable_shapes(a.data.shape, b.data.shape) == False:  
        #     raise Exception("Both args must have broadcastable sizes: {}, {}".format(a.shape, b.shape))

        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)

        # Create addition output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(np.divide(a.data, b.data), requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve forward inputs that we stored
        a, b = ctx.saved_tensors

        # calculate gradient of output w.r.t. each input
        grad_a = np.multiply(grad_output.data, 1/b.data)
        grad_b = np.multiply(grad_output.data, -np.divide(a.data, np.multiply(b.data,b.data)))

        if a.data.shape != grad_a.shape:
            grad_a = unbroadcast(grad_a, a.data.shape)
        if b.data.shape != grad_b.shape:
            grad_b = unbroadcast(grad_b, b.data.shape)

        # the order of gradients returned should match the order of the arguments
        return tensor.Tensor(grad_a), tensor.Tensor(grad_b)

class MatMul(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that both args are tensors
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))

        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)

        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data@b.data, requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve forward inputs that we stored
        a, b = ctx.saved_tensors

        # calculate gradient of output w.r.t. each input
        grad_a = grad_output.data@b.data.T
        grad_b = a.data.T@grad_output.data 

        # the order of gradients returned should match the order of the arguments
        return tensor.Tensor(grad_a), tensor.Tensor(grad_b)

class Sum(Function):
    @staticmethod
    def forward(ctx, a, axis, keepdims):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Only log of tensor is supported")
        ctx.axis = axis
        ctx.shape = a.shape
        if axis is not None:
            ctx.len = a.shape[axis]
        ctx.keepdims = keepdims
        requires_grad = a.requires_grad
        c = tensor.Tensor(a.data.sum(axis = axis, keepdims = keepdims), \
                          requires_grad=requires_grad, is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        grad = np.broadcast_to(grad_output.data, ctx.shape)
        assert grad.shape == ctx.shape

        # Take note that gradient tensors SHOULD NEVER have requires_grad = True.
        return tensor.Tensor(grad)

class Power(Function):
    @staticmethod
    def forward(ctx, a, p):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Only log of tensor is supported")

        ctx.save_for_backward(a)
        ctx.power = p

        requires_grad = a.requires_grad
        c = tensor.Tensor(np.power(a.data, p), requires_grad=requires_grad, is_leaf=not requires_grad)

        return c

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        p = ctx.power

        assert a.data.shape == grad_output.data.shape
        grad_a = p * np.power(a.data, p-1) * grad_output.data

        return tensor.Tensor(grad_a)

class Exp(Function):
    @staticmethod
    def forward(ctx, a):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Only log of tensor is supported")

        ctx.save_for_backward(a)

        requires_grad = a.requires_grad
        c = tensor.Tensor(np.exp(a.data), requires_grad=requires_grad, is_leaf=not requires_grad)

        return c

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]

        assert a.data.shape == grad_output.data.shape
        grad_a = np.exp(a.data) * grad_output.data

        return tensor.Tensor(grad_a)

class ReLU(Function):
    @staticmethod
    def forward(ctx, a):
        # Check that both args are tensors
        if not type(a).__name__ == 'Tensor':
            raise Exception("Both args must be Tensors: {}".format(type(a).__name__))

        ctx.save_for_backward(a)

        requires_grad = a.requires_grad

        #a_np = a.data
        #a_np_relu = np.piecewise(a_np, [a_np > 0, a_np <= 0], [lambda a_np: a_np, lambda a_np: 0])
        #c = tensor.Tensor(a_np_relu, requires_grad=requires_grad, is_leaf=not requires_grad)
        c = tensor.Tensor(a.data*(a.data>0), requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve forward inputs that we stored
        a = ctx.saved_tensors[0]

        # calculate gradient of output w.r.t. input
        #a_np = a.data
        #relu_prime = np.piecewise(a_np, [a_np > 0, a_np <= 0], [lambda a_np: 1, lambda a_np: 0])
        relu_prime = 1*(a.data>0)
        grad_a = np.multiply(grad_output.data, relu_prime)

        # the order of gradients returned should match the order of the arguments
        return tensor.Tensor(grad_a)

class Sigmoid(Function):
    @staticmethod
    def forward(ctx, a):
        b_data = np.divide(1.0, np.add(1.0, np.exp(-a.data)))
        ctx.out = b_data[:]
        b = tensor.Tensor(b_data, requires_grad=a.requires_grad)
        b.is_leaf = not b.requires_grad
        return b

    @staticmethod
    def backward(ctx, grad_output):
        b = ctx.out
        grad = grad_output.data * b * (1-b)
        return tensor.Tensor(grad)
    
class Tanh(Function):
    @staticmethod
    def forward(ctx, a):
        b = tensor.Tensor(np.tanh(a.data), requires_grad=a.requires_grad)
        ctx.out = b.data[:]
        b.is_leaf = not b.requires_grad
        return b

    @staticmethod
    def backward(ctx, grad_output):
        out = ctx.out
        grad = grad_output.data * (1-out**2)
        return tensor.Tensor(grad)

class Slice(Function):
    @staticmethod
    def forward(ctx,x,indices):
        '''
        Args:
            x (tensor): Tensor object that we need to slice
            indices (int,list,Slice): This is the key passed to the __getitem__ function of the Tensor object when it is sliced using [ ] notation.
        '''
        ctx.save_for_backward(x)
        ctx.indices = indices
        requires_grad = x.requires_grad

        return tensor.Tensor(x.data[indices], requires_grad=requires_grad, is_leaf=not requires_grad)

    @staticmethod
    def backward(ctx,grad_output):
        x = ctx.saved_tensors[0]
        indices = ctx.indices

        grad = np.zeros(x.shape)
        grad[indices] = grad_output.data

        return tensor.Tensor(grad)

class Cat(Function):
    @staticmethod
    def forward(ctx,*args):
        '''
        Args:
            args (list): [*seq, dim] 
        
        NOTE: seq (list of tensors) contains the tensors that we wish to concatenate while dim (int) is the dimension along which we want to concatenate 
        '''
        *seq, dim = args

        # First Tensor
        result = seq[0].data
        concat_idx = seq[0].shape[dim]
        dims = [concat_idx]

        # Concatenate Subsequent Tensors
        for i in range(1,len(seq)):
            result = np.concatenate((result,seq[i].data), axis=dim)
            concat_idx += seq[i].shape[dim]
            dims.append(concat_idx)

        ctx.dim = dim
        ctx.dims = dims
        requires_grad = seq[0].requires_grad

        return tensor.Tensor(result, requires_grad=requires_grad, is_leaf=not requires_grad)


    @staticmethod
    def backward(ctx,grad_output):
        dim = ctx.dim
        dims = ctx.dims

        result = np.split(grad_output.data, dims, axis=dim)
        for i in range(len(result)):
            result[i] = tensor.Tensor(result[i])

        return (*result, None)

class Dropout(Function):
    @staticmethod
    def forward(ctx, x, p=0.5, is_train=False):
        """Forward pass for dropout layer.

        Args:
            ctx (ContextManager): For saving variables between forward and backward passes.
            x (Tensor): Data tensor to perform dropout on
            p (float): The probability of dropping a neuron output.
                       (i.e. 0.2 -> 20% chance of dropping)
            is_train (bool, optional): If true, then the Dropout module that called this
                                       is in training mode (`<dropout_layer>.is_train == True`).
                                       
                                       Remember that Dropout operates differently during train
                                       and eval mode. During train it drops certain neuron outputs.
                                       During eval, it should NOT drop any outputs and return the input
                                       as is. This will also affect backprop correspondingly.
        """
        if not type(x).__name__ == 'Tensor':
            raise Exception("Only dropout for tensors is supported")
        ctx.is_train = is_train
        if is_train == True:
            mask = np.random.binomial(1, 1-p, x.shape) * (1/(1-p))
            ctx.mask = mask
            dropped_x = tensor.Tensor(np.multiply(x.data, mask), requires_grad=x.requires_grad, \
                                      is_leaf=not x.requires_grad)
            return dropped_x
        else:
            return x
        
    @staticmethod
    def backward(ctx, grad_output):
        if ctx.is_train == True:
            mask = ctx.mask
            grad = np.multiply(grad_output.data, mask)
            return tensor.Tensor(grad), None
        else:
            return grad_output

class Conv1d(Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride):
        """The forward/backward of a Conv1d Layer in the comp graph.
        
        Notes:
            - Make sure to implement the vectorized version of the pseudocode
            - See Lec 10 slides # TODO: FINISH LOCATION OF PSEUDOCODE
            - No, you won't need to implement Conv2d for this homework.
        
        Args:
            x (Tensor): (batch_size, in_channel, input_size) input data
            weight (Tensor): (out_channel, in_channel, kernel_size)
            bias (Tensor): (out_channel,)
            stride (int): Stride of the convolution
        
        Returns:
            Tensor: (batch_size, out_channel, output_size) output data
        """
        # For your convenience: ints for each size
        batch_size, in_channel, input_size = x.shape
        out_channel, _, kernel_size = weight.shape
        
        # TODO: Save relevant variables for backward pass
        ctx.save_for_backward(x, weight, bias)
        ctx.stride = stride
        
        # TODO: Get output size by finishing & calling get_conv1d_output_size()
        output_size = get_conv1d_output_size(input_size, kernel_size, stride)

        # TODO: Initialize output with correct size
        out = np.zeros((batch_size, out_channel, output_size))
        
        # TODO: Calculate the Conv1d output.
        # Remember that we're working with np.arrays; no new operations needed.
        # w.shape = (out_channel, in_channel x kernel_size)
        w = weight.data.reshape(out_channel, int(in_channel*kernel_size))
        for b in range(batch_size):
            for j in range(output_size):
                seg_idx = int(j*stride)
                # segment.shape = (in_channel x kernel_size, )
                segment = x.data[b,:,seg_idx:(seg_idx+kernel_size)].flatten()
                # out[b,:,j].shape = (out_channel, ) <- computing all out_channels at once
                out[b,:,j] = w@segment + bias.data

        # TODO: Put output into tensor with correct settings and return
        return tensor.Tensor(out, requires_grad=x.requires_grad, is_leaf=not x.requires_grad)
    
    @staticmethod
    def backward(ctx, grad_output):
        # TODO: Finish Conv1d backward pass. It's surprisingly similar to the forward pass.
        x, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        batch_size, in_channel, input_size = x.shape
        out_channel, _, kernel_size = weight.shape
        output_size = get_conv1d_output_size(input_size, kernel_size, stride)

        grad_x = np.zeros(x.shape)
        grad_weight = np.zeros(weight.shape)
        grad_bias = np.zeros(bias.shape)

        # # upsample grad_output with (stride-1) zeros in between each output element
        # zup = conv1d_upsample(grad_output.data, stride)

        # # zero pad grad_output with (kernel_size-1)
        # pad_width = kernel_size-1
        # zpad = np.pad(zup, pad_width, 'constant', constant_values=0)
        # # Since np.pad() pads in all dimensions, get rid of unnecessary pads
        # z_hat = zpad[pad_width:-pad_width,pad_width:-pad_width,:]

        # w_flipped = np.fliplr(weight.data.reshape(out_channel, int(in_channel*kernel_size)))
        # w = w_flipped.reshape(out_channel, in_channel, kernel_size)

        # for b in range(batch_size):
        #     for m in range(in_channel):
        #         for i in range(input_size):
        #             # z_hat_seg.shape = (out_channel x kernel_size, )
        #             z_hat_seg = z_hat[b,:,i:(i+kernel_size)].flatten()
        #             # w_flipped[:,m].flatten().shape = (out_channel x kernel_size, )
        #             grad_x[b, m, i] = z_hat_seg@w[:,m].flatten()

        for b in range(batch_size):
            for i in range(output_size):
                m = int(i*stride)
                for j in range(out_channel):
                    grad_bias[j] += grad_output.data[b,j,i]
                    for k in range(in_channel):
                        for l in range(kernel_size):
                            grad_x[b, k, m+l] += weight.data[j,k,l] * grad_output.data[b,j,i]
                            grad_weight[j,k,l] += grad_output.data[b,j,i] * x.data[b,k,m+l]

        return tensor.Tensor(grad_x), tensor.Tensor(grad_weight), tensor.Tensor(grad_bias)

def get_conv1d_output_size(input_size, kernel_size, stride):
    """Gets the size of a Conv1d output.

    Notes:
        - This formula should NOT add to the comp graph.
        - Yes, Conv2d would use a different formula,
        - But no, you don't need to account for Conv2d here.
        
        - If you want, you can modify and use this function in HW2P2.
            - You could add in Conv1d/Conv2d handling, account for padding, dilation, etc.
            - In that case refer to the torch docs for the full formulas.

    Args:
        input_size (int): Size of the input to the layer
        kernel_size (int): Size of the kernel
        stride (int): Stride of the convolution

    Returns:
        int: size of the output as an int (not a Tensor or np.array)
    """
    # TODO: implement the formula in the writeup. One-liner; don't overthink
    return ((input_size - kernel_size) // stride) + 1

def conv1d_upsample(dz, stride):
    batch_size, out_channel, output_size = dz.shape
    
    if stride > 1:
        dzup = np.zeros((batch_size, out_channel, int((output_size-1)*(stride+1))))
        for i in range(output_size):
            dzup[:,:,int(i*(stride))] = dz[:,:,i]    
    else:
        dzup = dz
    
    return dzup

def cross_entropy(predicted, target):
    """Calculates Cross Entropy Loss (XELoss) between logits and true labels.
    For MNIST, don't call this function directly; use nn.loss.CrossEntropyLoss instead.

    Args:
        predicted (Tensor): (batch_size, num_classes) logits
        target (Tensor): (batch_size,) true labels

    Returns:
        Tensor: the loss as a float, in a tensor of shape ()
    """
    batch_size, num_classes = predicted.shape

    # Tip: You can implement XELoss all here, without creating a new subclass of Function.
    #      However, if you'd prefer to implement a Function subclass you're free to.
    #      Just be sure that nn.loss.CrossEntropyLoss calls it properly.

    # Tip 2: Remember to divide the loss by batch_size; this is equivalent
    #        to reduction='mean' in PyTorch's nn.CrossEntropyLoss

    #raise Exception("TODO: Implement XELoss for comp graph")

    # Along each row of data get max of index (gives index of columns)
    max_i_vals = np.argmax(predicted.data, axis=1)
    assert max_i_vals.shape == target.shape
    a = tensor.Tensor(predicted.data[np.arange(batch_size), max_i_vals].reshape((batch_size, 1)))
    
    logSumExp = a + (predicted - a).exp().sum(axis=1, keepdims=True).log()
    log_softmax = predicted - logSumExp

    one_hot_target = to_one_hot(target, num_classes) # shape of batch_size x num_classes
    loss = (log_softmax * one_hot_target).sum()
    assert loss.shape == ()

    return tensor.Tensor(-1/batch_size) * loss

def to_one_hot(arr, num_classes):
    """(Freebie) Converts a tensor of classes to one-hot, useful in XELoss

    Example:
    >>> to_one_hot(Tensor(np.array([1, 2, 0, 0])), 3)
    [[0, 1, 0],
     [0, 0, 1],
     [1, 0, 0],
     [1, 0, 0]]
     
    Args:
        arr (Tensor): Condensed tensor of label indices
        num_classes (int): Number of possible classes in dataset
                           For instance, MNIST would have `num_classes==10`
    Returns:
        Tensor: one-hot tensor
    """
    arr = arr.data.astype(int)
    a = np.zeros((arr.shape[0], num_classes))
    a[np.arange(len(a)), arr] = 1

    return tensor.Tensor(a, requires_grad = True)