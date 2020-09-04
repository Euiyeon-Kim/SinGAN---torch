class Config:
    # [BASIC CONFIGS]
    mode = 'train'
    useGPU = True
    manualSeed = None
    img_channel = 3
    img_save_iter = 500

    # [DATA]
    img_path = 'Input/Images/balloons.png'
    generator_path = None                       # Saved generator path
    discriminator_path = None                   # Saved discriminator path

    # [PYRAMID PARAMETERS]
    scale_factor = 0.75                         # Pyramid scale factor pow(0.5, 1/6)
    min_size = 25                               # Image minimal size at the coarser scale
    max_size = 250                              # Image maximal size at the coarser scale
    noise_amp = 0.1                             # Additive noise cont weight

    # [NETWORK PARAMETERS]
    nfc = 32
    min_nfc = 32
    num_layers = 5
    kernel_size = 3
    stride = 1
    pad = 0                                     # Don't use layer padding for variation of the samples

    # [OPTIMIZATION PARAMETERS]
    num_iter = 2000                             # # of epochs(iteration) to train per scale
    alpha = 10                                  # Reconstruction loss weight
    gamma = 1e-1
    gp_lambda = 0.1
    g_lr = 5e-4
    d_lr = 5e-4
    beta1 = 0.5
    beta2 = 0.999
    n_critic = 3
    generator_iter = 3
