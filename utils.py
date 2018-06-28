import numpy as np
import keras.backend as K
from concise.utils.pwm import DEFAULT_BASE_BACKGROUND, pssm_array2pwm_array, _pwm2pwm_info
from concise.utils.plot import seqlogo, seqlogo_fig
from concise.eval_metrics import auc, auprc
import matplotlib.pyplot as plt


def evaluate(model, x, y, add_metrics={}):
    """Evaluate model performance
    
    Args:
      model: Keras model
      x: model inputs
      y: model targets
      add_metrics: additional metrics
      
    Returns:
      A dictionary of type: <metric>: <value>
    """
    keras_metrics = {name: metric 
                     for name, metric in zip(model.metrics_names, 
                                             model.evaluate(x, y, verbose=0))}
    y_pred = model.predict(x)
    add_metrics= {k: v(y, y_pred) for k,v in add_metrics.items()}
    return {**keras_metrics, **add_metrics}



def plot_filters(W, ncol=2, figsize=(10,10)):
    """Plot convolutional filters as motifs
    
    Args:
      weights: weights returned by `model.layers[0].get_weights()[0]`
      ncol: number of columns in the plot
      figsize: Matplotlib figure size (width, height)
    """
    N = W.shape[2]
    nrow = int(np.ceil(N/ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=figsize)
    for i in range(N):
        ax = fig.axes[i]
        seqlogo(W[:,:,i], ax=ax);
        ax.set_title(f"Filter: {i}")
    plt.tight_layout()
    
    
def plot_history(history, figsize=(8, 5)):
    """Plot metric values through training
    
    Args:
      history: object returned by model.fit(, callbacks=[History(), ...])
      metric: metric name from model.metrics_names
    """
    # Setup the figure object
    figsize=(figsize[0]*len(history.model.metrics_names), figsize[1])
    fig = plt.figure(figsize=figsize)
    
    for i, metric in enumerate(history.model.metrics_names):
        plt.subplot(1, len(history.model.metrics_names), i+1)
        
        plt.plot(history.epoch, history.history[metric], label=f'train')
        plt.plot(history.epoch, history.history['val_'+metric], label=f'valid')
        
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel(metric)
        plt.title(metric)
        
    plt.tight_layout()
    
    

def input_grad(model, x, layer_idx=-2):
    """Get the gradient of model output (before the final activation layer) w.r.t. model input
    
    Args:
      model: Sequence-based keras model
      x: one-hot-encoded DNA sequence
      layer_idx: output layer index
    """
    fn = K.function([model.input], K.gradients(model.layers[layer_idx].output, [model.input]))
    return fn([x])[0]


def plot_seq_importance(model, x, xlim=None, layer_idx=-2, figsize=(25, 3)):
    """Plot input x gradient sequence importance score
    
    Args:
      model: DNA-sequence based Sequential keras model
      x: one-hot encoded DNA sequence
      xlim: restrict the plotted xrange
      figsize: matplotlib figure size
    """
    seq_len = x.shape[1]
    if xlim is None:
        xlim = (0, seq_len)
    grads = input_grad(model, x, layer_idx=layer_idx)
    for i in range(len(x)):
        seqlogo_fig(grads[i]*x[i], figsize=figsize)
        plt.xticks(list(range(xlim[0], xlim[1], 5)))
        plt.xlim(xlim)
