# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

import warnings 
warnings.filterwarnings("ignore")

import argparse
import json
import os
import torch
from utils import build_model, get_lr, average_checkpoints, last_n_checkpoints
import time
import gc

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from models.loss import WaveFlowLossDataParallel
from mel2samp import Mel2Samp, MAX_WAV_VALUE
from scipy.io.wavfile import write

def parse_args(parser):
    """
    Parse commandline arguments.
    """    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')
    parser.add_argument('-w', '--warm_start', action='store_true',
                        help='warm start. i.e. load_state_dict() with strict=False and optimizer & scheduler are initialized.')
    parser.add_argument('-s', '--synthesize', action='store_true',
                        help='run synthesize loop only. does not train or evaluate the model.')
    parser.add_argument('-t', '--temp', type=float, default=1.,
                        help='temperature during synthesize loop. defaults to 1. only applicable if -s is specified')
    parser.add_argument('-a', '--average_checkpoint', type=int, default=0,
                        help='checkpoint averaging. averages the given number of latest checkpoints for synthesize.')
    parser.add_argument('-e', '--epsilon', type=float, default=None,
                        help='epsilon value for polyak averaging. only applied if -a > 0. defaults to None (plain averaging)')
    
    
    distributed = parser.add_argument_group('distributed setup')
    distributed.add_argument('--rank', default=0, type=int,
                             help='Rank of the process, do not set! Done by multiproc module')
    distributed.add_argument('--world-size', default=1, type=int,
                             help='Number of processes, do not set! Done by multiproc module')
    distributed.add_argument('--dist-url', type=str, default='tcp://localhost:23456',
                             help='Url used to set up distributed training')
    distributed.add_argument('--group-name', type=str, default='group_name',
                             required=False, help='Distributed group name')
    distributed.add_argument('--dist-backend', default='nccl', type=str, choices={'nccl'},
                             help='Distributed run backend')    

    args = parser.parse_args()
    
    
    


def load_checkpoint(checkpoint_path, model, optimizer, scheduler,fp16_run):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    iteration = checkpoint_dict['iteration']
    if optimizer is not None and scheduler is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
        scheduler.load_state_dict(checkpoint_dict['scheduler'])
    model_for_loading = checkpoint_dict['model']
    if fp16_run:
        amp.load_state_dict(checkpoint['amp_state'])
        
    # band-aid fix for h_cache, remove it
    if 'h_cache' in model_for_loading:
        del model_for_loading['h_cache']
    try:
        model.load_state_dict(model_for_loading)
    except RuntimeError:
        print("DataParallel weight detected. loading...")
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in model_for_loading.items():
            name = k.replace("module.", "")  # remove `module.`
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict)                                                
    print("Loaded checkpoint '{}' (iteration {})" .format(
        checkpoint_path, iteration))
    return model, optimizer, scheduler, iteration

def load_checkpoint_warm_start(checkpoint_path, model, optimizer, scheduler,fp16_run):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    iteration = checkpoint_dict['iteration']
    model_for_loading = checkpoint_dict['model']
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in model_for_loading.items() if
                       (k in model_dict) and (model_dict[k].shape == model_for_loading[k].shape)}
    model_dict.update(pretrained_dict)
    missing_and_unexpected_keys = model.load_state_dict(pretrained_dict, strict=False)
    if fp16_run:
        amp.load_state_dict(checkpoint['amp_state'])    
    print("WARNING: only part of the model loaded. below are missing and unexpected keys, make sure that they are correct:")
    print(missing_and_unexpected_keys)
    print("Loaded checkpoint '{}' (iteration {})".format(
        checkpoint_path, iteration))
    return model, optimizer, scheduler, iteration

def load_averaged_checkpoint_warm_start(checkpoint_path, model, optimizer, scheduler,fp16_run):
    # checkpoint_path is dir in this function
    assert os.path.isdir(checkpoint_path)
    list_checkpoints = last_n_checkpoints(checkpoint_path, args.average_checkpoint)
    iteration = 0
    model_for_loading = average_checkpoints(list_checkpoints, args.epsilon)['model']
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in model_for_loading.items() if
                       (k in model_dict) and (model_dict[k].shape == model_for_loading[k].shape)}
    model_dict.update(pretrained_dict)
    if fp16_run:
        amp.load_state_dict(checkpoint['amp_state'])    
    missing_and_unexpected_keys = model.load_state_dict(pretrained_dict, strict=False)
    print("WARNING: only part of the model loaded. below are missing and unexpected keys, make sure that they are correct:")
    print(missing_and_unexpected_keys)
    print("Loaded checkpoint '{}' (iteration {})".format(
        checkpoint_path, iteration))
    return model, optimizer, scheduler, iteration

def save_checkpoint(model, optimizer, scheduler, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()  # dataparallel case
    else:
        model_state_dict = model.state_dict()
               
    if fp16_run:
        amp_state = amp.state_dict()
    else:
        amp_state = None
                
    torch.save({'model': model_state_dict,
                'iteration': iteration,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'amp_state': amp_state,
                'learning_rate': learning_rate}, filepath)
    

# eval and synth functions
def evaluate(model, num_gpus, epoch, test_loader, output_directory, epochs, learning_rate, lr_decay_step, lr_decay_gamma,
          sigma, iters_per_checkpoint, batch_size, seed, fp16_run, checkpoint_path, with_tensorboard):
    # eval loop
    model.eval()
    epoch_eval_loss = 0

    for i, batch in enumerate(test_loader):
        with torch.no_grad():
            mel, audio = batch
            mel = torch.autograd.Variable(mel.cuda())
            audio = torch.autograd.Variable(audio.cuda())
            outputs = model(audio, mel)
            loss = criterion(outputs)
            if num_gpus > 1:
                reduced_loss = loss.mean().item()
            else:
                reduced_loss = loss.item()
            epoch_eval_loss += reduced_loss
         if i==0 :
            save_audio = audio[0]

    epoch_eval_loss = epoch_eval_loss / len(test_loader)
    print(" EVAL {}:\t{:.9f}".format(iteration, epoch_eval_loss))
    if with_tensorboard:
        logger.add_scalar('eval_loss', epoch_eval_loss, iteration)
        logger.flush()
        
        
    samples_directory        
    samples_directory = os.path.join(output_directory, "samples", waveflow_config["model_name"]) ) 
    audio_path = os.path.join(samples_directory, "generate_{}.wav".format(epoch))        
    write(audio_path, data_config["sampling_rate"], save_audio)
    
    return model.train()

def train(model, num_gpus, output_directory, epochs, learning_rate, lr_decay_step, lr_decay_gamma,
          sigma, iters_per_checkpoint, batch_size, seed, fp16_run,
          checkpoint_path, with_tensorboard):

    criterion = WaveFlowLossDataParallel(sigma)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma)

    if fp16_run:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    # Load checkpoint if one exists
    iteration = 0
    if checkpoint_path != "":
        if args.warm_start:
            print("INFO: --warm_start. optimizer and scheduler are initialized and strict=False for load_state_dict().")
            if args.average_checkpoint == 0:
                model, optimizer, scheduler, iteration = load_checkpoint_warm_start(checkpoint_path, model, optimizer, scheduler,fp16_run)
            else:
                print("INFO: --average_checkpoint > 0. loading an averaged weight of last {} checkpoints...".format(args.average_checkpoint))
                model, optimizer, scheduler, iteration = load_averaged_checkpoint_warm_start(checkpoint_path, model, optimizer, scheduler,fp16_run)
        else:
            model, optimizer, scheduler, iteration = load_checkpoint(checkpoint_path, model, optimizer, scheduler,fp16_run)
        iteration += 1  # next iteration is iteration + 1

    if distributed_run:
        model = DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
            find_unused_parameters=True)
        model = torch.nn.DataParallel(model)

    trainset = Mel2Samp("train", False, False, **data_config)
    testset = Mel2Samp("test", False, False, **data_config)
    
    
    if distributed_run:
        train_sampler, shuffle = DistributedSampler(trainset), False
        test_sampler,  shuffle = DistributedSampler(testset), False           
     else:
        train_sampler, shuffle = None, True   
        test_sampler,  shuffle = None, True          
             
    train_loader = DataLoader(trainset, num_workers=16, shuffle=shuffle,
                              sampler=train_sampler, batch_size=batch_size,
                              pin_memory=False,  drop_last=True)
    

    test_sampler = None
    test_loader = DataLoader(testset, num_workers=4, shuffle=False,
                             sampler=test_sampler,
                             batch_size=batch_size,
                             pin_memory=False,
                             drop_last=False)
    synthset = Mel2Samp("test", True, True, **data_config)
    synth_sampler = None



    if with_tensorboard:
        from tensorboardX import SummaryWriter
        logger = SummaryWriter(os.path.join(output_directory, waveflow_config["model_name"], 'logs'))

    model.train()
    epoch_offset = max(0, int(iteration / len(train_loader)))
    
    iter_size = len(train_loader)
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, epochs):
        tic_epoch = time.time()
        #print("Epoch: {}".format(epoch))
        
        for i, batch in enumerate(train_loader):
            tic_iter = time.time()

            model.zero_grad()

            mel, audio = batch
            mel = torch.autograd.Variable(mel.cuda())
            audio = torch.autograd.Variable(audio.cuda())
            outputs = model(audio, mel)

            loss = criterion(outputs)
            if num_gpus > 1:
                reduced_loss = loss.mean().item()
            else:
                reduced_loss = loss.item()

            if fp16_run:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.mean().backward()

            if fp16_run:
                grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 5.)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()

            toc_iter = time.time()
            dur_iter = toc_iter - tic_iter 
            
        
            print("{:5d}/{:5d}|{:12d}|{:6d} /{:6d} | loss \t{:.9f}, {:.4f}sec/iter".format(epoch,epochs, iteration, i, iter_size, reduced_loss, dur_iter), end='')
            if with_tensorboard:
                logger.add_scalar('training_loss', reduced_loss, i + len(train_loader) * epoch)
                logger.add_scalar('lr', get_lr(optimizer), i + len(train_loader) * epoch)
                logger.add_scalar('grad_norm', grad_norm, i + len(train_loader) * epoch)
                logger.flush()

            if (iteration % iters_per_checkpoint == 0):
                checkpoint_directory = os.path.join(output_directory, waveflow_config["model_name"])
                checkpoint_path = "{}/waveflow_{}".format(checkpoint_directory, iteration)
                save_checkpoint(model, optimizer, scheduler, learning_rate, iteration, checkpoint_path)

            iteration += 1
            scheduler.step()
            #### iter end 
        toc_epoch=time.time()
        dur_epoch = toc_epoch - tic_epoch
        print("{:.4f} sec/epoch".format(dur_epoch), end='')
        tic_eval=time.time()
        model = evaluate(model, num_gpus, epoch, test_loader,  **train_config)
        toc_eval = time.time()
        dur_eval = toc_eval - tic_eval 
        print(" eval {:.4f} sec/epoch".format(dur_eval), end='')
        #####epoch end 

        
main():    
    
    parser = argparse.ArgumentParser(description='PyTorch NanoFlow Training')
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()

    # Parse configs.  Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    train_config = config["train_config"]
    global data_config
    data_config = config["data_config"]
    global waveflow_config
    waveflow_config = config["waveflow_config"]
    
    if 'LOCAL_RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        local_rank = args.rank
        world_size = args.world_size

    distributed_run = world_size > 1    

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)   
    
    # Get shared output_directory ready
    if not os.path.isdir( os.path.join(output_directory)):
        os.makedirs(      os.path.join(output_directory), exist_ok=True)
        os.chmod(         os.path.join(output_directory), 0o775)    
    print("output directory", os.path.join(output_directory ) )        
    
    if not os.path.isdir( os.path.join(output_directory, waveflow_config["model_name"])):
        os.makedirs(      os.path.join(output_directory, waveflow_config["model_name"]), exist_ok=True)
        os.chmod(         os.path.join(output_directory, waveflow_config["model_name"]), 0o775)
    print("checkpoint directory", os.path.join(output_directory, waveflow_config["model_name"]) )
    
    # Get sample output_directory ready
    if not os.path.isdir( os.path.join(output_directory, "samples")):
        os.makedirs(      os.path.join(output_directory, "samples"), exist_ok=True)
        os.chmod(         os.path.join(output_directory, "samples"), 0o775)
        os.makedirs(      os.path.join(output_directory, "samples", waveflow_config["model_name"]), exist_ok=True)
        os.chmod(         os.path.join(output_directory, "samples", waveflow_config["model_name"]), 0o775)  
    print("sample directory", os.path.join(output_directory, "samples", waveflow_config["model_name"]) )    
         
    
    if distributed_run:
        init_distributed(args, world_size, local_rank, args.group_name)
        
    model = build_model(waveflow_config)
    
    
    train(model, num_gpus, **train_config)
    
    
 

if __name__ == "__main__":
    
    main( )

