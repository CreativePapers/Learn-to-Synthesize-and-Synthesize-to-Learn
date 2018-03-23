import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision import transforms
import numpy as np
import os
import time
import datetime
from attribute_transfer_model import Discriminator
from attribute_transfer_model import Encoder
from attribute_transfer_model import Decoder
from PIL import Image




class Solver(object):

    def __init__(self, face_data_loader, config):
        # Data loader
        self.face_data_loader = face_data_loader

        # Model parameters
        self.y_dim = config.y_dim
        self.num_layers=config.num_layers
        self.im_size = config.im_size
        self.g_first_dim = config.g_first_dim
        self.d_first_dim = config.d_first_dim
        self.enc_repeat_num = config.enc_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.d_train_repeat = config.d_train_repeat

        # Hyper-parameteres
        self.lambda_cls = config.lambda_cls
        self.lambda_id = config.lambda_id
        self.lambda_bi = config.lambda_bi
        self.lambda_gp = config.lambda_gp
        self.enc_lr = config.enc_lr
        self.dec_lr = config.dec_lr
        self.d_lr = config.d_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        # Training settings
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.batch_size = config.batch_size
        self.trained_model = config.trained_model

        # Test settings
        self.test_model = config.test_model

        # Path
        self.log_path = config.log_path
        self.sample_path = config.sample_path
        self.model_path = config.model_path
        self.test_path = config.test_path

        # Step size
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step

        # Set tensorboard
        self.build_model()
        self.use_tensorboard()

        # Start with trained model
        if self.trained_model:
            self.load_trained_model()

    def build_model(self):
        # Define encoder-decoder (generator) and a discriminator
        self.Enc = Encoder(self.g_first_dim, self.enc_repeat_num)
        self.Dec = Decoder(self.g_first_dim)
        self.D = Discriminator(self.im_size, self.d_first_dim, self.d_repeat_num)

        # Optimizers
        self.enc_optimizer = torch.optim.Adam(self.Enc.parameters(), self.enc_lr, [self.beta1, self.beta2])
        self.dec_optimizer = torch.optim.Adam(self.Dec.parameters(), self.dec_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])

        if torch.cuda.is_available():
            self.Enc.cuda()
            self.Dec.cuda()
            self.D.cuda()

    def load_trained_model(self):

        self.Enc.load_state_dict(torch.load(os.path.join(
            self.model_path, '{}_Enc.pth'.format(self.trained_model))))
        self.Dec.load_state_dict(torch.load(os.path.join(
            self.model_path, '{}_Dec.pth'.format(self.trained_model))))
        self.D.load_state_dict(torch.load(os.path.join(
            self.model_path, '{}_D.pth'.format(self.trained_model))))
        print('loaded models (step: {})..!'.format(self.trained_model))

    def use_tensorboard(self):
        from tensorboard_logger import Logger
        self.logger = Logger(self.log_path)

    def update_lr(self, enc_lr,dec_lr, d_lr):
        for param_group in self.enc_optimizer.param_groups:
            param_group['lr'] = enc_lr
        for param_group in self.dec_optimizer.param_groups:
            param_group['lr'] = dec_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset(self):
        self.enc_optimizer.zero_grad()
        self.dec_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def to_var(self, x, volatile=False):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, volatile=volatile)

    def calculate_accuracy(self, x, y):
        _, predicted = torch.max(x, dim=1)
        correct = (predicted == y).float()
        accuracy = torch.mean(correct) * 100.0
        return accuracy

    def denorm(self, x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def one_hot(self, labels, dim):
        """Convert label indices to one-hot vector"""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out


    def train(self):
        """Train attribute-guided face image synthesis model"""
        self.data_loader = self.face_data_loader
        # The number of iterations for each epoch
        iters_per_epoch = len(self.data_loader)

        sample_x = []
        sample_l=[]
        real_y = []
        for i, (images, landmark) in enumerate(self.data_loader):
            labels=images[1]
            sample_x.append(images[0])
            sample_l.append(landmark[0])
            real_y.append(labels)
            if i == 2:
                break

        # Sample inputs and desired domain labels for testing
        sample_x = torch.cat(sample_x, dim=0)
        sample_x = self.to_var(sample_x, volatile=True)
        sample_l = torch.cat(sample_l, dim=0)
        sample_l = self.to_var(sample_l, volatile=True)
        real_y = torch.cat(real_y, dim=0)

        sample_y_list = []
        for i in range(self.y_dim):
            sample_y = self.one_hot(torch.ones(sample_x.size(0)) * i, self.y_dim)
            sample_y_list.append(self.to_var(sample_y, volatile=True))

        # Learning rate for decaying
        d_lr = self.d_lr
        enc_lr=self.enc_lr
        dec_lr=self.dec_lr

        # Start with trained model
        if self.trained_model:
            start = int(self.trained_model.split('_')[0])
        else:
            start = 0

        # Start training
        start_time = time.time()
        for e in range(start, self.num_epochs):
            for i, (real_image, real_landmark) in enumerate(self.data_loader):
                #real_x: real image and real_l: conditional side image (landmark heatmap)
                real_x=real_image[0]
                real_label = real_image[1]
                real_l=real_landmark[0]

                # Sample fake labels randomly
                rand_idx = torch.randperm(real_label.size(0))
                fake_label = real_label[rand_idx]

                real_y = self.one_hot(real_label, self.y_dim)
                fake_y = self.one_hot(fake_label, self.y_dim)

                # Convert tensor to variable
                real_x = self.to_var(real_x)
                real_l = self.to_var(real_l)
                real_y = self.to_var(real_y)
                fake_y = self.to_var(fake_y)
                real_label = self.to_var(real_label)
                fake_label = self.to_var(fake_label)

                #================== Train Discriminator ================== #
                # Input images (original image+side images) are concatenated
                src_output, cls_output = self.D(torch.cat([real_x, real_l], 1))
                d_loss_real = - torch.mean(src_output)
                d_loss_cls = F.cross_entropy(cls_output, real_label)

                # Compute expression recognition accuracy on synthetic images
                if (i+1) % self.log_step == 0:
                    accuracies = self.calculate_accuracy(cls_output, real_label)
                    log = ["{:.2f}".format(acc) for acc in accuracies.data.cpu().numpy()]
                    print('Recognition Acc: ')
                    print(log)

                # Generate outputs and compute loss with fake generated images
                enc_feat = self.Enc(torch.cat([real_x, real_l], 1))
                fake_x, fake_l= self.Dec(enc_feat, fake_y)
                fake_x = Variable(fake_x.data)
                fake_l = Variable(fake_l.data)

                src_output, cls_output = self.D(torch.cat([fake_x, fake_l], 1))
                d_loss_fake = torch.mean(src_output)

                # Discriminator losses
                d_loss = self.lambda_cls * d_loss_cls+d_loss_real + d_loss_fake
                self.reset()
                d_loss.backward()
                self.d_optimizer.step()

                # Compute gradient penalty loss
                real=torch.cat([real_x, real_l], 1)
                fake=torch.cat([fake_x, fake_l], 1)
                alpha = torch.rand(real_x.size(0), 1, 1, 1).cuda().expand_as(real)
                interpolated = Variable(alpha * real.data + (1 - alpha) * fake.data, requires_grad=True)
                output, cls_output = self.D(interpolated)

                grad = torch.autograd.grad(outputs=output,
                                           inputs=interpolated,
                                           grad_outputs=torch.ones(output.size()).cuda(),
                                           retain_graph=True,
                                           create_graph=True,
                                           only_inputs=True)[0]

                grad = grad.view(grad.size(0), -1)
                grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
                d_loss_gp = torch.mean((grad_l2norm - 1)**2)

                # Gradient penalty loss
                d_loss = self.lambda_gp * d_loss_gp
                self.reset()
                d_loss.backward()
                self.d_optimizer.step()

                # Logging
                loss = {}
                loss['D/loss_real'] = d_loss_real.data[0]
                loss['D/loss_fake'] = d_loss_fake.data[0]
                loss['D/loss_cls'] = d_loss_cls.data[0]
                loss['D/loss_gp'] = d_loss_gp.data[0]

                # ================== Train Encoder-Decoder networks ================== #
                if (i+1) % self.d_train_repeat == 0:

                    # Original-to-target and target-to-original domain
                    enc_feat = self.Enc(torch.cat([real_x, real_l], 1))
                    fake_x, fake_l = self.Dec(enc_feat, fake_y)
                    src_output, cls_output=self.D(torch.cat([fake_x, fake_l], 1))
                    g_loss_fake = - torch.mean(src_output)

                    #rec_feat = self.Enc(fake_x)
                    rec_feat = self.Enc(torch.cat([fake_x, fake_l], 1))
                    rec_x,rec_l=self.Dec(rec_feat, real_y)

                    # bidirectional loss of the images
                    g_loss_rec_x = torch.mean(torch.abs(real_x - rec_x))
                    g_loss_rec_l=torch.mean(torch.abs(real_l-rec_l))

                    #bidirectional loss of the latent feature
                    g_loss_feature = torch.mean(torch.abs(enc_feat - rec_feat))

                    #identity loss of the images
                    g_loss_identity_x = torch.mean(torch.abs(real_x - fake_x))
                    g_loss_identity_l = torch.mean(torch.abs(real_l - fake_l))

                    # attribute classification loss for the fake generated images
                    g_loss_cls = F.cross_entropy(cls_output, fake_label)

                    # Backward + Optimize (generator (encoder-decoder) losses), we update decoder two times for each encoder update
                    g_loss = g_loss_fake +self.lambda_bi * g_loss_rec_x +self.lambda_bi * g_loss_rec_l +self.lambda_bi * g_loss_feature+self.lambda_id * g_loss_identity_x+self.lambda_id * g_loss_identity_l+self.lambda_cls * g_loss_cls
                    self.reset()
                    g_loss.backward()
                    self.enc_optimizer.step()
                    self.dec_optimizer.step()
                    self.dec_optimizer.step()

                    # Logging Generator losses
                    loss['G/loss_feature'] = g_loss_feature.data[0]
                    loss['G/loss_identity_x'] = g_loss_identity_x.data[0]
                    loss['G/loss_identity_l'] = g_loss_identity_l.data[0]
                    loss['G/loss_rec_x'] = g_loss_rec_x.data[0]
                    loss['G/loss_rec_l'] = g_loss_rec_l.data[0]
                    loss['G/loss_fake'] = g_loss_fake.data[0]
                    loss['G/loss_cls'] = g_loss_cls.data[0]

                # Print out log
                if (i+1) % self.log_step == 0:
                    elapsed = time.time() - start_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))

                    log = "Elapsed [{}], Epoch [{}/{}], Iter [{}/{}]".format(
                        elapsed, e+1, self.num_epochs, i+1, iters_per_epoch)

                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)


                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, e * iters_per_epoch + i + 1)

                # Synthesize images
                if (i+1) % self.sample_step == 0:
                    fake_image_list = [sample_x]
                    for sample_y in sample_y_list:
                        enc_feat = self.Enc(torch.cat([sample_x, sample_l], 1))
                        sample_result,sample_landmark = self.Dec(enc_feat, sample_y)
                        fake_image_list.append(sample_result)
                    fake_images = torch.cat(fake_image_list, dim=3)
                    save_image(self.denorm(fake_images.data),
                        os.path.join(self.sample_path, '{}_{}_fake.png'.format(e+1, i+1)),nrow=1, padding=0)
                    print('Generated images and saved into {}..!'.format(self.sample_path))


                # Save checkpoints
                if (i+1) % self.model_save_step == 0:
                    torch.save(self.Enc.state_dict(),
                        os.path.join(self.model_path, '{}_{}_Enc.pth'.format(e+1, i+1)))
                    torch.save(self.Dec.state_dict(),
                        os.path.join(self.model_path, '{}_{}_Dec.pth'.format(e+1, i+1)))
                    torch.save(self.D.state_dict(),
                        os.path.join(self.model_path, '{}_{}_D.pth'.format(e+1, i+1)))

            # Decay learning rate
            if (e+1) > (self.num_epochs - self.num_epochs_decay):
                d_lr -= (self.d_lr / float(self.num_epochs_decay))
                enc_lr-= (self.enc_lr / float(self.num_epochs_decay))
                dec_lr-=(self.dec_lr / float(self.num_epochs_decay))
                self.update_lr(enc_lr, dec_lr, d_lr)
                print ('Decay learning rate to enc_lr: {}, d_lr: {}.'.format(enc_lr, d_lr))



    def test(self):
        """Generating face images owning target attributes (desired expressions) """
        # Load trained models
        Enc_path = os.path.join(self.model_path, '{}_Enc.pth'.format(self.test_model))
        Dec_path = os.path.join(self.model_path, '{}_Dec.pth'.format(self.test_model))
        self.Enc.load_state_dict(torch.load(Enc_path))
        self.Dec.load_state_dict(torch.load(Dec_path))
        self.Enc.eval()
        self.Dec.eval()

        data_loader = self.face_data_loader

        for i, (real_image, real_landmark) in enumerate(data_loader):
            org_c = real_image[1]
            real_x = real_image[0]
            real_l = real_landmark[0]
            real_x = self.to_var(real_x, volatile=True)
            real_l = self.to_var(real_l, volatile=True)

            target_y_list = []
            for j in range(self.y_dim):
                target_y = self.one_hot(torch.ones(real_x.size(0)) * j, self.y_dim)
                target_y_list.append(self.to_var(target_y, volatile=True))

            # Target image generation
            fake_image_list = [real_x]
            for target_y in target_y_list:
                enc_feat = self.Enc(torch.cat([real_x, real_l], 1))
                sample_result, sample_landmark = self.Dec(enc_feat, target_y)
                fake_image_list.append(sample_result)
            fake_images = torch.cat(fake_image_list, dim=3)
            save_path = os.path.join(self.test_path, '{}_fake.png'.format(i+1))
            save_image(self.denorm(fake_images.data), save_path, nrow=1, padding=0)
            print('Generated images and saved into "{}"..!'.format(save_path))

