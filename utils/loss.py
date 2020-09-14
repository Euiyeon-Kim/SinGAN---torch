import torch


def calcul_gp(discriminator, real, fake, device, use_acm=True):
    alpha = torch.rand(1, 1)
    alpha = alpha.expand(real.size())
    alpha = alpha.to(device)

    interpolated = alpha * real + ((1 - alpha) * fake)
    interpolated = interpolated.to(device)
    interpolated = torch.autograd.Variable(interpolated, requires_grad=True)

    if use_acm:
        interpolated_prob_out, _, _, _ = discriminator(interpolated)
    else:
        interpolated_prob_out = discriminator(interpolated)

    gradients = torch.autograd.grad(outputs=interpolated_prob_out, inputs=interpolated,
                                    grad_outputs=torch.ones(interpolated_prob_out.size()).to(device),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp
