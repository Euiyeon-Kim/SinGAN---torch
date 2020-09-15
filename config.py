class Config:
    # [BASIC CONFIGS]
    mode = 'train_SR'
    use_acm = True
    useGPU = True
    manualSeed = None
    img_channel = 3
    img_save_iter = 500

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

    # [GENERATOR]
    generator_iter = 3
    rec_weights = 10                            # Reconstruction loss weight

    # [DISCRIMINATOR]
    n_critic = 3
    gp_weights = 0.1
    # [ACM]
    num_heads = 8
    use_acm_oth = False
    acm_weights = 1

    # [OPTIMIZATION PARAMETERS]
    num_iter = 2000                             # # of epochs(iteration) to train per scale
    milestones = [1600]
    gamma = 1e-1
    g_lr = 5e-4
    d_lr = 5e-4
    beta1 = 0.5
    beta2 = 0.999

    # [DATA]
    img_path = 'Input/Images/33039_LR.png'
    exp_dir = 'exp/test'                        # f'exp/balloons/scale-{scale_factor}_alp-{alpha}'
    generator_path = None                       # Saved generator path
    discriminator_path = None                   # Saved discriminator path

    # [Inference]
    use_fixed_noise = True
    save_all_pyramid = True
    save_attention_map = True
    gen_start_scale = 0
    scale_h = 1
    scale_w = 1
    num_samples = 10

    # [SR]
    sr_factor = 4
