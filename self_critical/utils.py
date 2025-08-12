import numpy as np
import torch
import torch.nn as nn
import tqdm
import os
from .cider.pyciderevalcap.ciderD.ciderD import CiderD
from .bleu.bleu import Bleu


def write2txt(fp, info, mode='a'):
    """
    Write information to a text file
    
    Args:
        fp: file path
        info: information to write
        mode: file mode ('a' for append, 'w' for write)
    """
    with open(fp, mode=mode) as f:
        f.write(info)
        f.write('\n')


def _array_to_str(arr, sos_token, eos_token):
    """
    Convert array of tokens to string representation
    
    Args:
        arr: array of token indices (can be numpy array, list, or scalar)
        sos_token: start of sentence token
        eos_token: end of sentence token
    
    Returns:
        str: string representation of tokens
    """
    # Handle scalar case (single token)
    if np.isscalar(arr):
        return str(arr)
    
    # Handle array case
    out = ''
    for i in range(len(arr)):
        if arr[i] == eos_token:
            break
        out += str(arr[i]) + ' '
    out += str(eos_token)  # optional
    return out.strip()


def get_4_bleu_score(output, ground_truth, reward_scorer, sos_token=1, eos_token=2):
    """
    Calculate BLEU-1,2,3,4 scores for CE baseline evaluation
    
    Args:
        output: decoder output predictions [batch_size, seq_len, vocab_size]
        ground_truth: ground truth sentences [batch_size, seq_len]
        reward_scorer: BLEU scorer instance
        sos_token: start of sentence token (default=1)
        eos_token: end of sentence token (default=2)
    
    Returns:
        tuple: (bleu1_score, bleu2_score, bleu3_score, bleu4_score)
    """
    device = output.device
    batch_size = output.size(0)
    
    # Convert output probabilities to predicted tokens
    predicted_tokens = torch.argmax(output, dim=-1)  # [batch_size, seq_len]
    
    # Convert to numpy for evaluation
    predicted_tokens = predicted_tokens.cpu().numpy()
    ground_truth = ground_truth.cpu().numpy()
    
    # Prepare data for BLEU calculation
    sample_result = []
    gts = {}
    
    for i in range(batch_size):
        # Convert predicted tokens to string
        pred_str = _array_to_str(predicted_tokens[i], sos_token, eos_token)
        sample_result.append({'image_id': i, 'caption': [pred_str]})
        
        # Convert ground truth to string
        gt_str = _array_to_str(ground_truth[i], sos_token, eos_token)
        gts[i] = [gt_str]
    
    # Calculate BLEU scores using the reward_scorer
    if isinstance(reward_scorer, Bleu):
        _, scores_mat = reward_scorer.compute_score(gts, sample_result)
        
        # Extract individual BLEU scores
        scores_b1 = np.array(scores_mat[0]).mean()
        scores_b2 = np.array(scores_mat[1]).mean()
        scores_b3 = np.array(scores_mat[2]).mean()
        scores_b4 = np.array(scores_mat[3]).mean()
        
        return scores_b1, scores_b2, scores_b3, scores_b4
    else:
        print("Warning: reward_scorer is not a BLEU scorer")
        return 0.0, 0.0, 0.0, 0.0


def get_cider_score(output, ground_truth, reward_scorer, sos_token=1, eos_token=2):
    """
    Calculate CIDEr score for CE baseline evaluation
    
    Args:
        output: decoder output predictions [batch_size, seq_len, vocab_size]
        ground_truth: ground truth sentences [batch_size, seq_len]
        reward_scorer: CIDEr scorer instance
        sos_token: start of sentence token (default=1)
        eos_token: end of sentence token (default=2)
    
    Returns:
        float: cider_score
    """
    device = output.device
    batch_size = output.size(0)
    
    # Convert output probabilities to predicted tokens
    predicted_tokens = torch.argmax(output, dim=-1)  # [batch_size, seq_len]
    
    # Convert to numpy for evaluation
    predicted_tokens = predicted_tokens.cpu().numpy()
    ground_truth = ground_truth.cpu().numpy()
    
    # Prepare data for CIDEr calculation
    sample_result = []
    gts = {}
    
    for i in range(batch_size):
        # Convert predicted tokens to string
        pred_str = _array_to_str(predicted_tokens[i], sos_token, eos_token)
        sample_result.append({'image_id': i, 'caption': [pred_str]})
        
        # Convert ground truth to string
        gt_str = _array_to_str(ground_truth[i], sos_token, eos_token)
        gts[i] = [gt_str]
    
    # Calculate CIDEr score using the reward_scorer
    if isinstance(reward_scorer, CiderD):
        _, scores = reward_scorer.compute_score(gts, sample_result)
        cider_score = np.array(scores).mean()
        return cider_score
    else:
        print("Warning: reward_scorer is not a CIDEr scorer")
        return 0.0

def log_training_metrics(epoch, loss, bleu1, bleu2, bleu3, bleu4, log_path):
    """
    Log training metrics to text file
    
    Args:
        epoch: current epoch
        loss: training loss
        bleu1, bleu2, bleu3, bleu4: BLEU scores
        log_path: path to log file
    """
    # Create directory if it doesn't exist
    log_dir = os.path.dirname(log_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    info = f'epoch:{epoch} loss:{loss:.6f} bleu1:{bleu1:.4f} bleu2:{bleu2:.4f} bleu3:{bleu3:.4f} bleu4:{bleu4:.4f}'
    write2txt(log_path, info)


def log_training_metrics_cider(epoch, loss, cider_score, log_path):
    """
    Log training metrics with CIDEr score to text file
    
    Args:
        epoch: current epoch
        loss: training loss
        cider_score: CIDEr score
        log_path: path to log file
    """
    # Create directory if it doesn't exist
    log_dir = os.path.dirname(log_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    info = f'epoch:{epoch} loss:{loss:.6f} cider:{cider_score:.4f}'
    write2txt(log_path, info)


def log_rl_training_metrics(epoch, loss, advantage_mean, reward_mean, bleu1, bleu2, bleu3, bleu4, log_path):
    """
    Log RL training metrics to text file
    
    Args:
        epoch: current epoch
        loss: training loss
        advantage_mean: mean advantage value
        reward_mean: mean reward value
        bleu1, bleu2, bleu3, bleu4: BLEU scores
        log_path: path to log file
    """
    # Create directory if it doesn't exist
    log_dir = os.path.dirname(log_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    info = f'epoch:{epoch} loss:{loss:.6f} advantage:{advantage_mean:.4f} reward:{reward_mean:.4f} bleu1:{bleu1:.4f} bleu2:{bleu2:.4f} bleu3:{bleu3:.4f} bleu4:{bleu4:.4f}'
    write2txt(log_path, info)


def log_rl_training_metrics_cider(epoch, loss, advantage_mean, reward_mean, cider_score, log_path):
    """
    Log RL training metrics with CIDEr score to text file
    
    Args:
        epoch: current epoch
        loss: training loss
        advantage_mean: mean advantage value
        reward_mean: mean reward value
        cider_score: CIDEr score
        log_path: path to log file
    """
    # Create directory if it doesn't exist
    log_dir = os.path.dirname(log_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    info = f'epoch:{epoch} loss:{loss:.6f} advantage:{advantage_mean:.4f} reward:{reward_mean:.4f} cider:{cider_score:.4f}'
    write2txt(log_path, info)


def get_ciderd_scorer_europarl(split_captions, test_data_num, sos_token, eos_token):

    all_caps = np.concatenate((split_captions, test_data_num))
    print('====> get_ciderd_scorer begin, seeing {} sentences'.format(len(all_caps)))

    refs_idxs = []
    for caps in all_caps:
        ref_idxs = []
        ref_idxs.append(_array_to_str(caps, sos_token, eos_token))
        refs_idxs.append(ref_idxs)

    scorer = CiderD(refs_idxs)
    del refs_idxs
    del ref_idxs
    print('====> get_ciderd_scorer end')
    return scorer

def get_bleu_scorer_europarl(n=4):

    scorer = Bleu(n=n)
    print('====> get_bleu_scorer end')

    return scorer


def get_self_critical_reward_sc(sample_captions, fns, ground_truth,
                             sos_token, eos_token, scorer):
    # the first dim of fns are the same with samples. fns is a list.
    device = sample_captions.device
    batch_size = len(ground_truth)
    seq_per_img = len(fns) // batch_size
    sample_captions = sample_captions.cpu().numpy()
    ground_truth = ground_truth.cpu().numpy()

    max_seq_len = sample_captions.shape[1]
    sample_result = []
    gts = {}
    # first multiple samples.
    for fn in fns:
        sample_result.append({'image_id': fn, 'caption': [_array_to_str(sample_captions[fn], sos_token, eos_token)]})
        caps = []
        caps.append(_array_to_str(ground_truth[fn//seq_per_img][:max_seq_len], sos_token, eos_token))
        gts[fn] = caps

    if isinstance(scorer, CiderD):
        _, scores = scorer.compute_score(gts, sample_result)  # [bs*5,1]
        scores = torch.from_numpy(scores).to(device).view(-1, seq_per_img)  # [bs,5]
        detailed_reward = None
    elif isinstance(scorer, Bleu):
        _, scores_mat = scorer.compute_score(gts, sample_result)
        scores_b1 = np.array(scores_mat[0]).mean()
        scores_b2 = np.array(scores_mat[1]).mean()
        scores_b3 = np.array(scores_mat[2]).mean()
        scores_b4 = np.array(scores_mat[3]).mean()
        detailed_reward = (scores_b1, scores_b2, scores_b3, scores_b4)
        scores = (np.array(scores_mat[0]) + np.array(scores_mat[3]))/2
        scores = torch.from_numpy(scores).to(device).view(-1, seq_per_img)  # [bs,5]

    scores.requires_grad = False
    baseline = (scores.sum(1, keepdim=True) - scores) / (scores.shape[1] - 1) # [bs,5]
    scores = scores - baseline # [bs,5]
    scores = scores.view(-1, 1)  # [bs*5, 1]
    return scores, baseline.mean(), detailed_reward


def get_self_critical_reward_newsc_TXRL(sample_captions, fns, ground_truth,
                                        sos_token, eos_token, scorer):
    # the first dim of fns are the same with samples. fns is a list.
    device = sample_captions.device
    batch_size = len(ground_truth)
    seq_per_img = len(fns) // batch_size
    sample_captions = sample_captions.cpu().numpy()
    ground_truth = ground_truth.cpu().numpy()

    max_seq_len = sample_captions.shape[1]
    sample_result = []
    gts = {}
    # first multiple samples.
    for fn in fns:
        sample_result.append({'image_id': fn, 'caption': [_array_to_str(sample_captions[fn], sos_token, eos_token)]})
        caps = []
        caps.append(_array_to_str(ground_truth[fn // seq_per_img][:max_seq_len], sos_token, eos_token))
        gts[fn] = caps

    _, scores = scorer.compute_score(gts, sample_result)  # [bs*5,1]
    scores = torch.from_numpy(scores).to(device).view(-1, seq_per_img)  # [bs,5]
    scores.requires_grad = False
    if seq_per_img > 1:
        baseline = (scores.sum(1, keepdim=True) - scores) / (scores.shape[1] - 1)  # [bs,5]
        scores = scores - baseline  # [bs,5]
    scores = scores.view(-1, 1)  # [bs*5, 1]

    if seq_per_img > 1:
        return scores, baseline.mean()
    else:
        return scores, scores.mean()