import functools
import math
import numpy as np
from tqdm import tqdm, trange


import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
import torchvision

# Import my stuff
import inception_utils
import utils
import losses

def prepare_parser():
  usage = 'Parser for all scripts.'
  parser = ArgumentParser(description=usage)

  parser.add_argument(
    '--model', type=str, default='BigGAN',
    help='Name of the model module (default: %(default)s)')


#model = __import__(config['model'])
#experiment_name = (config['experiment_name'] if config['experiment_name']
 #                  else utils.name_from_config(config))
#print('Experiment name is %s' % experiment_name)

#config['model'] if config['model'] != 'BigGAN' else None,
##
#D = model.Discriminator(**config).to(device)


#print('Loading weights...')
#utils.load_weights(G, D, state_dict,
#                   config['weights_root'], experiment_name, 
#                   config['load_weights'] if config['load_weights'] else None,
#                   G_ema if config['ema'] else None)

class G_D(nn.Module):
  def __init__(self, G, D):
    super(G_D, self).__init__()
    self.G = G
    self.D = D


  def forward(self, z, gy, x=None, dy=None, train_G=False, return_G_z=False,
              split_D=False):
    D_fake = self.D(G_z, gy)
    return D_fake

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
  with open(path, 'rb') as f:
    img = Image.open(f)
    return img.convert('RGB')

    
def main():
  # parse command line and run
  parser = utils.prepare_parser()
  parser = utils.add_sample_parser(parser)
  config = vars(parser.parse_args())
  print(config)
  run(config)
  # model = __import__(config['model'])
  # experiment_name = (config['experiment_name'] if config['experiment_name']
  #                  else utils.name_from_config(config))
  # print('Experiment name is %s' % experiment_name)
  # D = model.Discriminator(**config).to(device)
  # G = model.Generator(**config).to(device)
  
  # state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
  #               'best_IS': 0, 'best_FID': 999999, 'config': config}
  # if config['ema']:
  #   print('Preparing EMA for G with decay of {}'.format(config['ema_decay']))
  #   G_ema = model.Generator(**{**config, 'skip_init':True, 
  #                              'no_optim': True}).to(device)
  #   ema = utils.ema(G, G_ema, config['ema_decay'], config['ema_start'])
  # else:
  #   G_ema, ema = None, None

  # print('Loading weights...')
  # utils.load_weights(G, D, state_dict,
  #                  config['weights_root'], experiment_name, 
  #                  config['load_weights'] if config['load_weights'] else None,
  #                  G_ema if config['ema'] else None, strict=False, load_optim=False)

  #img = pil_loader("~/new_arch/a/BigGAN-romanesque-islamicdisc/romanesque_1260.jpg")
  
  #images, labels = utils.sample()
 # D_fake = D(images,labels)

def run(config):
  # Prepare state dict, which holds things like epoch # and itr #
  state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                'best_IS': 0, 'best_FID': 999999, 'config': config}
                
  # Optionally, get the configuration from the state dict. This allows for
  # recovery of the config provided only a state dict and experiment name,
  # and can be convenient for writing less verbose sample shell scripts.
  if config['config_from_name']:
    utils.load_weights(None, None, state_dict, config['weights_root'], 
                       config['experiment_name'], config['load_weights'], None,
                       strict=False, load_optim=False)
    # Ignore items which we might want to overwrite from the command line
    for item in state_dict['config']:
      if item not in ['z_var', 'base_root', 'batch_size', 'G_batch_size', 'use_ema', 'G_eval_mode']:
        config[item] = state_dict['config'][item]
  # Update the config dict as necessary
  # This is for convenience, to add settings derived from the user-specified
  # configuration into the config-dict (e.g. inferring the number of classes
  # and size of the images from the dataset, passing in a pytorch object
  # for the activation specified as a string)
  config['resolution'] = utils.imsize_dict[config['dataset']]
  config['n_classes'] = utils.nclass_dict[config['dataset']]
  config['G_activation'] = utils.activation_dict[config['G_nl']]
  config['D_activation'] = utils.activation_dict[config['D_nl']]
  # By default, skip init if resuming training.
  if config['resume']:
    print('Skipping initialization for training resumption...')
    config['skip_init'] = True
  config = utils.update_config_roots(config)
  device = 'cuda'
  
  # Seed RNG
  utils.seed_rng(config['seed'])

  # Prepare root folders if necessary
  utils.prepare_root(config)

  # Setup cudnn.benchmark for free speed
  torch.backends.cudnn.benchmark = True

  # Import the model--this line allows us to dynamically select different files.
  model = __import__(config['model'])
  experiment_name = (config['experiment_name'] if config['experiment_name']
                       else utils.name_from_config(config))
  print('Experiment name is %s' % experiment_name)

  # Next, build the model
  G = model.Generator(**config).to(device)
  D = model.Discriminator(**config).to(device)
  utils.count_parameters(G)
  
  # Load weights
  print('Loading weights...')
  # Here is where we deal with the ema--load ema weights or load normal weights
  utils.load_weights(G if not (config['use_ema']) else None, None, state_dict, 
                     config['weights_root'], experiment_name, config['load_weights'],
                     G if config['ema'] and config['use_ema'] else None,
                     strict=False, load_optim=False)
  # Update batch size setting used for G
  G_batch_size = max(config['G_batch_size'], config['batch_size']) 
  z_, y_ = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'],
                             device=device, fp16=config['G_fp16'], 
                             z_var=config['z_var'])
  
  if config['G_eval_mode']:
    print('Putting G in eval mode..')
    G.eval()
  else:
    print('G is in %s mode...' % ('training' if G.training else 'eval'))
    
  #Sample function
  sample = functools.partial(utils.sample, G=G, z_=z_, y_=y_, config=config)  
  if config['accumulate_stats']:
    print('Accumulating standing stats across %d accumulations...' % config['num_standing_accumulations'])
    utils.accumulate_standing_stats(G, z_, y_, config['n_classes'],
                                    config['num_standing_accumulations'])
    
  
  # Sample a number of images and save them to an NPZ, for use with TF-Inception
  if config['sample_npz']:
    # Lists to hold images and labels for images
    x, y = [], []
    print('Sampling %d images and saving them to npz...' % config['sample_num_npz'])
    for i in trange(int(np.ceil(config['sample_num_npz'] / float(G_batch_size)))):
      with torch.no_grad():
        images, labels = sample()
      x += [np.uint8(255 * (images.cpu().numpy() + 1) / 2.)]
      y += [labels.cpu().numpy()]
    x = np.concatenate(x, 0)[:config['sample_num_npz']]
    y = np.concatenate(y, 0)[:config['sample_num_npz']]    
    print('Images shape: %s, Labels shape: %s' % (x.shape, y.shape))
    npz_filename = '%s/%s/samples.npz' % (config['samples_root'], experiment_name)
    print('Saving npz to %s...' % npz_filename)
    np.savez(npz_filename, **{'x' : x, 'y' : y})
  
  # Prepare sample sheets
  if config['sample_sheets']:
    print('Preparing conditional sample sheets...')
    utils.sample_sheet(G, classes_per_sheet=utils.classes_per_sheet_dict[config['dataset']], 
                         num_classes=config['n_classes'], 
                         samples_per_class=10, parallel=config['parallel'],
                         samples_root=config['samples_root'], 
                         experiment_name=experiment_name,
                         folder_number=config['sample_sheet_folder_num'],
                         z_=z_,)
  # Sample interp sheets
  if config['sample_interps']:
    print('Preparing interp sheets...')
    for fix_z, fix_y in zip([False, False, True], [False, True, False]):
      utils.interp_sheet(G, num_per_sheet=16, num_midpoints=8,
                         num_classes=config['n_classes'], 
                         parallel=config['parallel'], 
                         samples_root=config['samples_root'], 
                         experiment_name=experiment_name,
                         folder_number=config['sample_sheet_folder_num'], 
                         sheet_number=0,
                         fix_z=fix_z, fix_y=fix_y, device='cuda')
  # Sample random sheet
  if config['sample_random']:
    print('Preparing random sample sheet...')
    images, labels = sample()
    print("labels size", images[1,:,:,:].shape)    
    torchvision.utils.save_image(images.float(),
                                 '%s/%s/random_samples.jpg' % (config['samples_root'], experiment_name),
                                 nrow=int(G_batch_size**0.5),
                                 normalize=True)

   # If using EMA, prepare it
  if config['ema']:
    print('Preparing EMA for G with decay of {}'.format(config['ema_decay']))
    G_ema = model.Generator(**{**config, 'skip_init':True, 
                               'no_optim': True}).to(device)
    ema = utils.ema(G, G_ema, config['ema_decay'], config['ema_start'])
  else:
    G_ema, ema = None, None
  
  # FP16?
  if config['G_fp16']:
    print('Casting G to float16...')
    G = G.half()
    if config['ema']:
      G_ema = G_ema.half()
  if config['D_fp16']:
    print('Casting D to fp16...')
    D = D.half()
    # Consider automatically reducing SN_eps?
  GD = model.G_D(G, D)
  #print(G)
  #print(D)
  print('Number of params in G: {} D: {}'.format(
    *[sum([p.data.nelement() for p in net.parameters()]) for net in [G,D]]))
  # Prepare state dict, which holds things like epoch # and itr #
  state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                'best_IS': 0, 'best_FID': 999999, 'config': config}

  # If loading from a pre-trained model, load weights
  if config['resume']:
    print('Loading weights...')
    utils.load_weights(G, D, state_dict,
                       config['weights_root'], experiment_name, 
                       config['load_weights'] if config['load_weights'] else None,
                       G_ema if config['ema'] else None)

  # If parallel, parallelize the GD module
  if config['parallel']:
    GD = nn.DataParallel(GD)
    if config['cross_replica']:
      patch_replication_callback(GD)

  
  D_fake = D(images,labels)
  print("D_fake ",D_fake)
if __name__ == '__main__':
  main()
