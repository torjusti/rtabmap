#! /usr/bin/env python3
#
# Drop this file in the root folder of SuperGlue git: https://github.com/magicleap/SuperGluePretrainedNetwork
# To use with rtabmap:
#   --Vis/CorNNType 6 --SuperGlue/Path "~/SuperGluePretrainedNetwork/rtabmap_superglue.py"
#

import random
import numpy as np
import torch

#import sys
#import os
#print(os.sys.path)
#print(sys.version)

from models.matching import SuperGlue

torch.set_grad_enabled(False)

device = 'cpu'
superglue = []

def init(descriptorDim, matchThreshold, iterations, cuda, model):
    print("SuperGlue python init()")
    # Load the SuperGlue model.
    global device
    device = 'cuda' if torch.cuda.is_available() and cuda else 'cpu'
    assert model == "indoor" or model == "outdoor", "Available models for SuperGlue are 'indoor' or 'outdoor'"
    config = {
        'superglue': {
            'weights': model,
            'sinkhorn_iterations': iterations,
            'match_threshold': matchThreshold,
            'descriptor_dim' : descriptorDim
        }
    }
    global superglue
    superglue = SuperGlue(config.get('superglue', {})).eval().to(device)

def match(kptsFrom, kptsTo, scoresFrom, scoresTo, descriptorsFrom, descriptorsTo, imageWidth, imageHeight):
    #print("SuperGlue python match()")
    global device
    kptsFrom = np.asarray(kptsFrom)
    kptsFrom = kptsFrom[None, :, :]
    kptsTo = np.asarray(kptsTo)
    kptsTo = kptsTo[None, :, :]
    scoresFrom = np.asarray(scoresFrom)
    scoresFrom = scoresFrom[None, :]
    scoresTo = np.asarray(scoresTo)
    scoresTo = scoresTo[None, :]
    descriptorsFrom = np.transpose(np.asarray(descriptorsFrom))
    descriptorsFrom = descriptorsFrom[None, :, :]
    descriptorsTo = np.transpose(np.asarray(descriptorsTo))
    descriptorsTo = descriptorsTo[None, :, :]
      
    data = {
       # SuperGlue only reads the shape of the images (for keypoint normalization)
       'image0': torch.empty(1, 1, imageHeight, imageWidth, device=device),
       'image1': torch.empty(1, 1, imageHeight, imageWidth, device=device),
       'scores0': torch.from_numpy(scoresFrom).to(device),
       'scores1': torch.from_numpy(scoresTo).to(device),
       'keypoints0': torch.from_numpy(kptsFrom).to(device),
       'keypoints1': torch.from_numpy(kptsTo).to(device),
       'descriptors0': torch.from_numpy(descriptorsFrom).to(device),
       'descriptors1': torch.from_numpy(descriptorsTo).to(device),
    }
    

    global superglue
    # grad mode is thread-local in torch: the module-level
    # torch.set_grad_enabled(False) doesn't apply when called from another
    # thread, and keeping gradients would hold all intermediate activations
    with torch.no_grad():
        results = superglue(data)

    matches0 = results['matches0'].to('cpu').numpy()
  
    matchesFrom = np.nonzero(matches0!=-1)[1]
    matchesTo = matches0[np.nonzero(matches0!=-1)]
       
    matchesArray = np.stack((matchesFrom, matchesTo), axis=1);
    
    # rtabmap expects format:
    #   matches: array Nx2 (type=9 or uint64)
    return matchesArray

def match_batch(kptsFrom, kptsTo, scoresFrom, scoresTo, descriptorsFrom, descriptorsTo, counts, imageWidth, imageHeight):
    # Batched version of match(): all arrays have a leading batch dimension B,
    # padded with zeros to the largest keypoint count in the batch. counts is
    # a (B, 2) int array with the true (from, to) keypoint counts of each pair.
    # Returns a (M, 3) int array of (batchIndex, matchFrom, matchTo).
    global device
    global superglue

    kptsFrom = torch.from_numpy(np.asarray(kptsFrom, dtype=np.float32)).to(device)             # (B, N0, 2)
    kptsTo = torch.from_numpy(np.asarray(kptsTo, dtype=np.float32)).to(device)                 # (B, N1, 2)
    scoresFrom = torch.from_numpy(np.asarray(scoresFrom, dtype=np.float32)).to(device)         # (B, N0)
    scoresTo = torch.from_numpy(np.asarray(scoresTo, dtype=np.float32)).to(device)             # (B, N1)
    descriptorsFrom = torch.from_numpy(np.asarray(descriptorsFrom, dtype=np.float32)).permute(0, 2, 1).contiguous().to(device)  # (B, D, N0)
    descriptorsTo = torch.from_numpy(np.asarray(descriptorsTo, dtype=np.float32)).permute(0, 2, 1).contiguous().to(device)      # (B, D, N1)
    counts = np.asarray(counts)

    data = {
        # SuperGlue only reads the shape of the images (for keypoint
        # normalization), the batch dimension is not used.
        'image0': torch.empty(1, 1, imageHeight, imageWidth, device=device),
        'image1': torch.empty(1, 1, imageHeight, imageWidth, device=device),
        'scores0': scoresFrom,
        'scores1': scoresTo,
        'keypoints0': kptsFrom,
        'keypoints1': kptsTo,
        'descriptors0': descriptorsFrom,
        'descriptors1': descriptorsTo,
    }

    try:
        with torch.no_grad():  # grad mode is thread-local, see match()
            results = superglue(data)
    except Exception:
        # free cached GPU memory so that the caller's fallback
        # (matching the pairs one by one) can proceed
        del data
        del kptsFrom, kptsTo, scoresFrom, scoresTo, descriptorsFrom, descriptorsTo
        torch.cuda.empty_cache()
        raise
    matches0 = results['matches0'].to('cpu').numpy()  # (B, N0)

    out = []
    for b in range(matches0.shape[0]):
        m = matches0[b]
        matchesFrom = np.nonzero(m != -1)[0]
        matchesTo = m[matchesFrom]
        # drop matches involving padded (fake) keypoints
        valid = (matchesFrom < counts[b, 0]) & (matchesTo < counts[b, 1])
        matchesFrom = matchesFrom[valid]
        matchesTo = matchesTo[valid]
        batchIndex = np.full(matchesFrom.shape, b)
        out.append(np.stack((batchIndex, matchesFrom, matchesTo), axis=1))

    if len(out):
        return np.concatenate(out, axis=0).astype(np.int64)
    return np.zeros((0, 3), dtype=np.int64)


if __name__ == '__main__':
    #test
    init(256, 0.2, 20, True, 'indoor')
    match([[1, 2], [1,3]], [[1, 3], [1,2]], [1, 3], [1,3], np.full((2, 256), 1),np.full((2, 256), 1), 640, 480)
