import os
from argparse import Namespace

from tqdm import tqdm
import time
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
import sys

sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from utils.common import tensor2im, log_input_image
from options.test_options import TestOptions
from models.psp import pSp


def run():
    test_opts = TestOptions().parse()

    out_path_results = os.path.join(test_opts.exp_dir, 'inference_results')
    out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled')

    os.makedirs(out_path_results, exist_ok=True)
    os.makedirs(out_path_coupled, exist_ok=True)

    # update test options with options used during training
    ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    opts.update(vars(test_opts))
    if 'learn_in_w' not in opts:
        opts['learn_in_w'] = False
    if 'output_size' not in opts:
        opts['output_size'] = 1024
    opts = Namespace(**opts)

    net = pSp(opts)
    net.eval()
    net.cuda()

    print('Loading dataset for {}'.format(opts.dataset_type))
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    opts.resize_factors = test_opts.resize_factors
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    dataset = InferenceDataset(root=opts.data_path,
                               transform=transforms_dict['transform_inference'],
                               opts=opts)
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=int(opts.test_workers),
                            drop_last=True)

    if opts.n_images is None:
        opts.n_images = len(dataset)

    latent_list = []
    for iter_time in range(test_opts.n_outputs_to_generate) :
        iter_out_path_results = out_path_results
        iter_out_path_coupled = out_path_coupled
        global_i = 0
        global_time = []
        for input_batch in dataloader:
            with torch.no_grad():
                input_cuda = input_batch.cuda().float()
                tic = time.time()
                result_batch, latent = run_on_batch(input_cuda, net, opts, iter_time)
                toc = time.time()
                global_time.append(toc - tic)
                latent_list.append(latent)

            result = tensor2im(result_batch[0])
            im_path = dataset.paths[0]

            if opts.couple_outputs or global_i % 100 == 0:
                input_im = log_input_image(input_batch[0], opts)
                resize_amount = (256, 256) if opts.resize_outputs else (opts.output_size, opts.output_size)
                if opts.resize_factors is not None:
                    # for super resolution, save the original, down-sampled, and output
                    source = Image.open(im_path)
                    res = np.concatenate([np.array(source.resize(resize_amount)),
                                        np.array(input_im.resize(resize_amount, resample=Image.NEAREST)),
                                        np.array(result.resize(resize_amount))], axis=1)
                else:
                    # otherwise, save the original and output
                    res = np.concatenate([np.array(input_im.resize(resize_amount)),
                                        np.array(result.resize(resize_amount))], axis=1)
                Image.fromarray(res).save(os.path.join(iter_out_path_coupled, f'{iter_time}.jpg'))

            im_save_path = os.path.join(iter_out_path_results, f'{iter_time}.jpg')
            Image.fromarray(np.array(result)).save(im_save_path)

            global_i += 1
            break
        #stats_path = os.path.join(opts.exp_dir, 'stats.txt')
    result_str = 'Runtime {:.4f}+-{:.4f}'.format(np.mean(global_time), np.std(global_time))
    print(result_str)
    latents = torch.cat(latent_list, dim=0)
    print(torch.mean(latents, dim=0))
    latent_std = torch.std(latents, dim=0)
    print(latent_std.sum(dim=-1))

    #with open(stats_path, 'w') as f:
    #    f.write(result_str)


def run_on_batch(inputs, net, opts, seed):
    torch.manual_seed(seed)
    if opts.latent_mask is None:
        result_batch, latent = net(inputs, randomize_noise=False, resize=opts.resize_outputs, return_latents=True)
    else:
        latent_mask = [int(l) for l in opts.latent_mask.split(",")]
        result_batch = []
        for image_idx, input_image in enumerate(inputs):
            # get latent vector to inject into our input image
            vec_to_inject = np.random.randn(1, 512).astype('float32')
            _, latent_to_inject = net(torch.from_numpy(vec_to_inject).to("cuda"),
                                      input_code=True,
                                      return_latents=True)
            # get output image with injected style vector
            res, latent = net(input_image.unsqueeze(0).to("cuda").float(),
                      latent_mask=latent_mask,
                      inject_latent=latent_to_inject,
                      alpha=opts.mix_alpha,
                      resize=opts.resize_outputs,
                      return_latents = True)
            result_batch.append(res)
        result_batch = torch.cat(result_batch, dim=0)
    return result_batch, latent


if __name__ == '__main__':
    run()
