import argparse
from tqdm import tqdm
import torch
import os
from DFCNN import AcousticModel
from readdata import SpeechData
import torch.nn as nn
import logging
from torchsummary import summary
import torchsnooper
import horovod.torch as hvd
from tensorboardX import SummaryWriter as writer
import random
from BeamSearch import ctcBeamSearch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
# required args
parser.add_argument('--data_type',
                    type=str,
                    required=True,
                    help='Either aishell or thchs')
parser.add_argument('--model_path',
                    type=str,
                    required=True,
                    help="loading or saving model")

# parameters for features.py and data iterator
parser.add_argument('--config_path',
                    type=str,
                    default='data_config',
                    help='config file')
parser.add_argument('--window_size',
                    type=int,
                    default=400,
                    help="Hamming window size, default value as 400")
parser.add_argument('--numcep',
                    type=int,
                    default=26,
                    help="MFCC feature numbers : number of cepstrum")
parser.add_argument('--time_window',
                    type=int,
                    default=25,
                    help="time window")
parser.add_argument('--step',
                    type=int,
                    default=10,
                    help="window shift step")
parser.add_argument('--batch_size',
                    type=int,
                    default=8,)
parser.add_argument('--max_window',
                    type=int,
                    default=1600,
                    help='Maximum of windows')
parser.add_argument('--max_seq_length',
                    type=int,
                    default=64,
                    help='maximum of sequence length')
# training procedures
parser.add_argument('--model_name',
                    type=str,
                    default="logs_am.pt")
parser.add_argument('--gpu_rank',
                    type=int,
                    default=4,
                    help="define number of gpus to use")
parser.add_argument('--lr',
                    type=float,
                    default=0.0001,
                    help="learning rate")
parser.add_argument('--epochs',
                    type=int,
                    default=5,
                    help="training epoch")
parser.add_argument('--save_step',
                    type=int,
                    default=10)
parser.add_argument('--load_model',
                    action="store_true")
parser.add_argument('--dev_step',
                    type=int,
                    default=5)
args = parser.parse_args()

if __name__ == '__main__':
    # initialize horovod
    hvd.init()

    from torch.utils.data import DataLoader
    import torch.optim as optim
    torch.cuda.set_device(hvd.local_rank())
    if torch.cuda.current_device() == 0:
        logger.info("Args are as : %s" % args)
    if torch.cuda.is_available():
        assert args.gpu_rank <= torch.cuda.device_count(),\
            "Should ensure that gpu_rank <= the number of yournvidia devices"
        logger.info("%d gpu(s) detected and using %d devices." % (torch.cuda.device_count(), args.gpu_rank))
    else:
        logger.info("Using cpu will take you quite a long time, not recommended!")
    train_data = SpeechData('data_config', dataset=args.data_type)
    dev_data = SpeechData('data_config', type='dev', dataset=args.data_type)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data,
                                                                    num_replicas=hvd.size(),
                                                                    rank=hvd.rank())
    dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_data,
                                                                  num_replicas=hvd.size(),
                                                                  rank=hvd.rank())

    vocab_size = train_data.label_nums()
    logger.info("Total PNY labels : %d" % vocab_size)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, sampler=train_sampler)
    dev_loader = DataLoader(dev_data, batch_size=int(args.batch_size/2), sampler=dev_sampler)

    model = AcousticModel(vocab_size=vocab_size, input_dimension=200)

    model_path = args.model_path + '/' + args.model_name
    if os.path.exists(model_path) and args.load_model:
        logger.info("Model: %s exists, loading existing model..." % model_path)
        model = torch.load(model_path)
    else:
        if not os.path.exists(args.model_path):
            os.mkdir(args.model_path)
        logger.info('No existing model in your provided path.')
    model = model.cuda()

    if torch.cuda.current_device() == 0:
        print(model)

    criterion = nn.CTCLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr * hvd.size()/2)

    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

    if torch.cuda.current_device() == 0:
        writer = writer()

    tr_run = 0
    for epoch in tqdm(range(args.epochs)):
        if (epoch+1) % args.dev_step == 0:
            model.eval()
            model.cuda()
            eval_loss = 0.0
            if torch.cuda.current_device() == 0:
                logger.info("Starting evaluating!")
            for batch_idx, samples in enumerate(dev_loader):
                # optimizer.zero_grad()
                X, y, input_lengths, label_lengths, transcripts = samples
                X = X.type(torch.FloatTensor).cuda()
                y = y.type(torch.LongTensor).cuda()
                input_lengths = model.convert(input_lengths)
                label_lengths = label_lengths.cuda()
                input_lengths = input_lengths.cuda()

                with torch.no_grad():
                    outputs = model(X)
                # print(outputs.shape, y.shape)
                # outputs = outputs.cpu()
                loss = criterion(outputs,
                                 y,
                                 input_lengths,
                                 label_lengths)
                eval_loss += loss
                # loss.backward()
                # optimizer.step()
                if (batch_idx + 1) % int(len(dev_data)/(args.gpu_rank*int(args.batch_size/2))) == 0:
                    # print(outputs)
                    logger.info("Evaluating step %d, step loss : %.5f , total_mean_loss : %.5f"
                                % (batch_idx + 1, loss, eval_loss / (batch_idx + 1)))
            # logger.info("mean_eval_loss: %.5f" % eval_loss/len(dev_data))
        else:
            model.train()
            model.cuda()
            train_sampler.set_epoch(epoch)
            epoch_loss = 0.0
            logger.info("Running epoch %d/%d =>" % (epoch+1, args.epochs))
            for batch_idx, samples in enumerate(train_loader):
                tr_run += 1
                optimizer.zero_grad()
                X, y, input_lengths, label_lengths, transcripts = samples
                X = X.type(torch.FloatTensor).cuda()
                # y = y.reshape((args.batch_size*64,))
                y = y.type(torch.LongTensor).cuda()
                input_lengths = model.convert(input_lengths)
                label_lengths = label_lengths.cuda()
                input_lengths = input_lengths.cuda()

                outputs = model(X)
                # print(outputs.shape, y.shape)
                # outputs = outputs.cpu()
                loss = criterion(outputs,
                                 y,
                                 input_lengths,
                                 label_lengths)
                if torch.cuda.current_device() == 0:
                    writer.add_scalar('tr_loss', loss.data.item(), tr_run)
                epoch_loss += loss

                loss.backward()
                optimizer.step()
                if (batch_idx+1) % 10 == 0 and torch.cuda.current_device() == 0:
                    # print(outputs)
                    logger.info("Training step %d, progress %d/%d, step loss : %.5f , total_mean_loss : %.5f"
                                % (batch_idx+1, batch_idx+1, int(len(train_data)/(args.batch_size * args.gpu_rank)),
                                    loss, epoch_loss/(batch_idx+1)))
                    # if loss < 10:
                    #     # select_index = random.randint(0, args.batch_size-1)
                    #     result = outputs.detach().transpose(0, 1).cpu().numpy()
                    #     # result = np.transpose(result, [1, 0, 2])
                    #     # print(result.shape)
                    #     input_length = input_lengths.cpu().numpy()
                    #     # result = np.hstack((result[:, 1:], result[:, 0].reshape((-1, 1))))
                    #     # classes = list(train_data.i2w.values())[1:]
                    #     decode = K.ctc_decode(result, input_length, greedy=False)
                    #     print(K.get_value(decode[0][0]))

        if (epoch+1) % args.save_step == 0 and torch.cuda.current_device() == 0:
            logger.info("Saving model parameters into %s" % model_path)
            torch.save(model.cpu(), model_path)

    writer.close()







