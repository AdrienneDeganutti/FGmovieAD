import torch
import torch.distributed as dist
import os
import os.path as op
import sys
import time
import json
from apex import amp
from apex.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from datetime import datetime, timedelta

pythonpath = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
print(pythonpath)
sys.path.insert(0, pythonpath)

from src.configs import get_cfg_defaults
from src.AVL_config.config import shared_configs, restore_training_settings, basic_check_arguments
from lib.data import make_data_loader
from lib.utils.config import init_config, Config
from lib.utils.logger import setup_logger_wandb as setup_logger
from lib.utils.metric_logger import MetricLogger
from src.utils.comm import dist_init, get_rank, is_main_process, get_world_size
from src.utils.load_save import TrainingRestorer, TrainingSaver
from src.utils.logger import TB_LOGGER, RunningMeter, add_log_to_file
from src.modeling.load_swin import get_swin_model, reload_pretrained_swin
from src.modeling.load_bert import get_bert_model
from src.modeling.video_captioning_e2e_vid_swin_bert import VideoTransformer
from src.utils.miscellaneous import NoOp, str_to_bool, set_seed
from src.solver import AdamW, WarmupLinearLR


def get_custom_args(base_config):
    parser = base_config.parser
    parser.add_argument('--max_num_frames', type=int, default=32)
    parser.add_argument('--img_res', type=int, default=224)
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument("--grid_feat",
                        type=str_to_bool,
                        nargs='?',
                        const=True,
                        default=True)
    parser.add_argument("--kinetics",
                        type=str,
                        default='400',
                        help="400 or 600")
    parser.add_argument("--att_mode",
                        type=str,
                        default='default',
                        help="default, full")
    parser.add_argument("--lambda_",
                        type=float,
                        default=0.5,
                        help="lambda_ loss")
    parser.add_argument("--pretrained_2d",
                        type=str_to_bool,
                        nargs='?',
                        const=True,
                        default=False)
    parser.add_argument("--vidswin_size", type=str, default='base')
    parser.add_argument('--freeze_backbone',
                        type=str_to_bool,
                        nargs='?',
                        const=True,
                        default=False)
    parser.add_argument('--freeze_passt',
                        type=str_to_bool,
                        nargs='?',
                        const=True,
                        default=False)
    parser.add_argument('--use_checkpoint',
                        type=str_to_bool,
                        nargs='?',
                        const=True,
                        default=False)
    parser.add_argument('--backbone_coef_lr', type=float, default=0.001)
    parser.add_argument("--reload_pretrained_swin",
                        type=str_to_bool,
                        nargs='?',
                        const=True,
                        default=False)
    parser.add_argument('--learn_mask_enabled',
                        type=str_to_bool,
                        nargs='?',
                        const=True,
                        default=False)
    parser.add_argument('--loss_sparse_w', type=float, default=0)
    parser.add_argument('--sparse_mask_soft2hard',
                        type=str_to_bool,
                        nargs='?',
                        const=True,
                        default=False)
    parser.add_argument(
        '--transfer_method',
        type=int,
        default=0,
        help=
        "0: load all SwinBERT pre-trained weights, 1: load only pre-trained sparse mask"
    )
    parser.add_argument(
        '--att_mask_expansion',
        type=int,
        default=0,
        help=
        "-1: random init, 0: random init and then diag-based copy, 1: interpolation"
    )
    parser.add_argument('--resume_checkpoint', type=str, default='None')
    parser.add_argument('--test_video_fname', type=str, default='None')
    args = base_config.parse_args()
    return args


def mixed_precision_init(args, model):
    max_iter = args.max_iter
    max_global_step = args.max_global_step
    global_iters_per_epoch = args.global_iters_per_epoch

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    decay_param_tp = [(n, p) for n, p in param_optimizer
                      if not any(nd in n for nd in no_decay)]
    no_decay_param_tp = [(n, p) for n, p in param_optimizer
                         if any(nd in n for nd in no_decay)]

    decay_swin_param_tp = [(n, p) for n, p in decay_param_tp if "swin." in n]
    decay_bert_param_tp = [(n, p) for n, p in decay_param_tp
                           if "swin." not in n]

    no_decay_swin_param_tp = [(n, p) for n, p in no_decay_param_tp
                              if "swin." in n]
    no_decay_bert_param_tp = [(n, p) for n, p in no_decay_param_tp
                              if "swin." not in n]

    weight_decay = 0.2
    coef_lr = args.backbone_coef_lr
    optimizer_grouped_parameters = [{
        'params': [p for n, p in decay_swin_param_tp],
        'weight_decay':
        weight_decay,
        'lr':
        args.learning_rate * coef_lr
    }, {
        'params': [p for n, p in decay_bert_param_tp],
        'weight_decay':
        weight_decay
    }, {
        'params': [p for n, p in no_decay_swin_param_tp],
        'weight_decay':
        0.0,
        'lr':
        args.learning_rate * coef_lr
    }, {
        'params': [p for n, p in no_decay_bert_param_tp],
        'weight_decay':
        0.0
    }]

    if args.mixed_precision_method == "fairscale":
        from fairscale.optim.oss import OSS
        optimizer = OSS(params=optimizer_grouped_parameters,
                        optim=AdamW,
                        lr=args.learning_rate,
                        eps=args.adam_epsilon)
    else:
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=args.learning_rate,
                          eps=args.adam_epsilon)
    if args.scheduler == "warmup_linear":
        scheduler = WarmupLinearLR(optimizer,
                                   max_global_step,
                                   warmup_ratio=args.warmup_ratio)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=int(max_iter /
                                                                  2.0),
                                                    gamma=0.1)

    if args.mixed_precision_method == "deepspeed":
        config = get_deepspeed_config(args)
        model, optimizer, _, _ = deepspeed.initialize(config_params=config,
                                                      model=model,
                                                      optimizer=optimizer,
                                                      lr_scheduler=scheduler)
    elif args.mixed_precision_method == "fairscale":
        from fairscale.optim.grad_scaler import ShardedGradScaler
        scaler = ShardedGradScaler()
        # this is equivalent to deepspeed zero_opt_stage = 2
        from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
        model = ShardedDDP(
            model,
            optimizer,
            reduce_buffer_size=0
            if args.fairscale_fp16 else 2**23,  # 2 ** 23 is the default value
            reduce_fp16=args.fairscale_fp16)
    else:
        # opt_level is O0, Apex will run as fp32
        model, optimizer = amp.initialize(model,
                                          optimizer,
                                          enabled=True,
                                          opt_level=f'O{args.amp_opt_level}')
        if args.distributed:  #
            model = DDP(model)
    return args, model, optimizer, scheduler


def check_arguments(args):
    # shared basic checks
    basic_check_arguments(args)
    # additional sanity check:
    args.max_img_seq_length = int(
        (args.max_num_frames / 2) * (int(args.img_res) / 32) *
        (int(args.img_res) / 32)) + 473

    if args.freeze_backbone or args.backbone_coef_lr == 0:
        args.backbone_coef_lr = 0
        args.freeze_backbone = True

    if 'reload_pretrained_swin' not in args.keys():
        args.reload_pretrained_swin = False

    if not len(args.pretrained_checkpoint) and args.reload_pretrained_swin:
        logger.info(
            "No pretrained_checkpoint to be loaded, disable --reload_pretrained_swin"
        )
        args.reload_pretrained_swin = False

    if args.learn_mask_enabled == True:
        args.attn_mask_type = 'learn_vid_att'


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, -1)[1].data  # argmax
    return logits == labels


def train(MAD_args, AVL_args, train_dataloader, val_dataloader, model, tokenizer,
          training_saver, optimizer, scheduler):
    
    meters = MetricLogger(delimiter='  ')
    max_iter = AVL_args.max_iter
    max_global_step = AVL_args.max_global_step
    global_iters_per_epoch = AVL_args.global_iters_per_epoch

    eval_log = []
    best_score = 0
    start_training_time = time.time()
    end = time.time()
    log_start = time.time()
    running_loss = RunningMeter('train_loss')
    running_batch_acc = RunningMeter('train_batch_acc')

    if AVL_args.restore_ratio > 0:
        restorer = TrainingRestorer(AVL_args, model, optimizer)
        global_step = restorer.global_step
    else:
        global_step = 0

    TB_LOGGER.global_step = global_step
    if not is_main_process() or AVL_args.restore_ratio <= 0:
        restorer = NoOp()

    training_saver.save_args(AVL_args)
    training_saver.save_tokenizer(tokenizer)

    for iteration, (img_keys, batch, meta_data) in enumerate(train_dataloader):
        iteration += 1
        data_time = time.time() - end
        batch = tuple(t.to(AVL_args.device) for t in batch)
        model.train()
        # img_feats (B, #F, C, W, H)
        inputs = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'token_type_ids': batch[2],
            'img_feats': batch[3],
            'masked_pos': batch[5],
            'masked_ids': batch[6],
            'input_token_ids': batch[7],
            'output_token_ids': batch[8],
        }

        if iteration == 1:
            for k, v in inputs.items():
                logger.info(f'{k} = {v.shape}')

        if AVL_args.deepspeed_fp16:
            # deepspeed does not autocast inputs
            inputs = fp32_to_fp16(inputs)

        if AVL_args.mixed_precision_method == "fairscale":
            with torch.cuda.amp.autocast(enabled=True):
                outputs = model(**inputs)
        else:
            outputs = model(**inputs)
        loss, logits = outputs[:2]

        if AVL_args.learn_mask_enabled:
            loss_sparsity = outputs[-1]
            loss = loss + (loss_sparsity * AVL_args.loss_sparse_w)

        lm_loss, mask_loss = outputs[2], outputs[3]

        batch_score = compute_score_with_logits(logits,
                                                inputs['output_token_ids'])
        batch_acc = torch.mean(batch_score.float())

        if AVL_args.learn_mask_enabled:
            loss_dict = {
                'loss': loss,
                'loss_sparsity': loss_sparsity.item(),
                'acc': batch_acc,
                'lm_loss': lm_loss,
                'mask_loss': mask_loss
            }
        else:
            loss_dict = {
                'loss': loss,
                'acc': batch_acc,
                'lm_loss': lm_loss,
                'mask_loss': mask_loss
            }
        meters.update(**loss_dict)
        running_loss(loss.item())
        running_batch_acc(batch_acc.item())

        # backward pass
        backward_now = iteration % AVL_args.gradient_accumulation_steps == 0
        if AVL_args.mixed_precision_method == "deepspeed":
            model.backward(loss)
        elif AVL_args.mixed_precision_method == "fairscale":
            scaler.scale(loss).backward()
        else:
            # apex
            with amp.scale_loss(loss,
                                optimizer,
                                delay_unscale=not backward_now) as scaled_loss:
                scaled_loss.backward()
        if backward_now:
            global_step += 1
            TB_LOGGER.add_scalar('train/loss', running_loss.val, global_step)

            lr_VisBone = optimizer.param_groups[0]["lr"]
            lr_LM = optimizer.param_groups[1]["lr"]

            TB_LOGGER.add_scalar("train/lr_lm", lr_LM, global_step)
            TB_LOGGER.add_scalar("train/ls_visBone", lr_VisBone, global_step)

            if AVL_args.max_grad_norm != -1:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer), AVL_args.max_grad_norm)
                TB_LOGGER.add_scalar("train/grad_norm", grad_norm, global_step)
            TB_LOGGER.step()
            if AVL_args.mixed_precision_method == "deepspeed":
                model.step()
            elif AVL_args.mixed_precision_method == "fairscale":
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                model.zero_grad()
            else:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            restorer.step()

            TB_LOGGER.add_scalar("train/loss", running_loss.val, global_step)
            TB_LOGGER.add_scalar("train/acc", running_batch_acc.val, global_step)
            log_start = time.time()

        batch_time = time.time() - end

        if backward_now:
            if global_step % AVL_args.logging_steps == 0 or global_step == max_global_step:
                if 'time_info' in meters.meters:
                    avg_time = meters.meters['time_info']['compute'].global_avg
                    eta_seconds = avg_time * (max_iter - iteration)
                    eta_string = str(
                        datetime.timedelta(seconds=int(eta_seconds)))
                else:
                    eta_string = 'Unknown'
                eta_seconds = batch_time * (max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                speed = AVL_args.num_gpus * AVL_args.logging_steps * len(
                    batch[0]) / (time.time() - log_start)
                memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
                logger.info(
                    meters.delimiter.join([
                        f"eta: {eta_string}",
                        f"iter: {iteration}",
                        f"global_step: {global_step}",
                        f'speed: {speed:.1f} images/sec',
                        f"{meters}",
                        f"lr (Visual Encoder): {lr_VisBone:.2e}",
                        f"lr (LM): {lr_LM:.2e}",
                        f"max mem: {memory:.0f}",
                    ]))
                TB_LOGGER.add_scalar("train/speed", speed, global_step)
                TB_LOGGER.add_scalar("train/memory", memory, global_step)
                TB_LOGGER.add_scalar("train/batch_time", batch_time, global_step)
                TB_LOGGER.add_scalar("train/data_time", data_time, global_step)
                log_start = time.time()

            if (AVL_args.save_steps > 0 and global_step % AVL_args.save_steps == 0
                ) or global_step == max_global_step or global_step == 1:
                epoch = global_step // global_iters_per_epoch

                checkpoint_dir = op.join(
                    AVL_args.output_dir,
                    'checkpoint-{}-{}'.format(epoch, global_step))
                if get_world_size() > 1:
                    dist.barrier()
                training_saver.save_model(checkpoint_dir, global_step, model,
                                          optimizer)
                if get_world_size() > 1:
                    dist.barrier()
                if AVL_args.evaluate_during_training:
                    logger.info(
                        f"Perform evaluation at iteration {iteration}, global_step {global_step}"
                    )
                    evaluate_file = evaluate(AVL_args, val_dataloader, model,
                                             tokenizer, checkpoint_dir)
                    if get_world_size() > 1:
                        dist.barrier()
                    if is_main_process():
                        with open(evaluate_file, 'r') as f:
                            res = json.load(f)
                        val_log = {f'valid/{k}': v for k, v in res.items()}
                        TB_LOGGER.log_scalar_dict(val_log)
                        # aml_run.log(name='CIDEr', value=float(res['CIDEr']))

                        best_score = max(best_score, res['CIDEr'])
                        res['epoch'] = epoch
                        res['iteration'] = iteration
                        res['best_CIDEr'] = best_score
                        eval_log.append(res)
                        with open(
                                op.join(
                                    AVL_args.output_dir,
                                    AVL_args.val_yaml.replace('/', '_') +
                                    'eval_logs.json'), 'w') as f:
                            json.dump(eval_log, f)
                    if get_world_size() > 1:
                        dist.barrier()

        if iteration > 2:
            meters.update(
                batch_time=batch_time,
                data_time=data_time,
            )
        end = time.time()

        if global_step >= max_global_step and (max_iter - iteration):
            logger.info(
                f'Missing {max_iter - iteration} iterations, early break')
            break

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        f'Total training time: {total_time_str} ({(total_training_time / max_iter):.4f} s / iter)'
    )

    return checkpoint_dir


def main(MAD_args, AVL_args):

    AVL_args.device = torch.device(AVL_args.device)
    dist_init(AVL_args)
    check_arguments(AVL_args)
    set_seed(AVL_args.seed, AVL_args.num_gpus)

    if AVL_args.mixed_precision_method == 'apex':
        fp16_training = f"apex 0{AVL_args.amp_opt_level}"

    logger.info("Device: {}, n_gpu: {}, rank: {}, "
                "16-bits training: {}".format(AVL_args.device, AVL_args.num_gpus, 
                                              get_rank(), fp16_training))

    if not is_main_process():
        logger.disabled = True
        training_saver = NoOp()
    else:
        training_saver = TrainingSaver(AVL_args.output_dir)
        TB_LOGGER.create(op.join(AVL_args.output_dir, 'log'))
        add_log_to_file(op.join(AVL_args.output_dir, 'log', "log.txt"))

    # Get Video Swin model
    swin_model = get_swin_model(AVL_args)
    # Get BERT and tokenizer
    bert_model, config, tokenizer = get_bert_model(AVL_args)
    # build SwinBERT based on training configs
    vl_transformer = VideoTransformer(AVL_args, config, swin_model, bert_model)
    vl_transformer.freeze_backbone(freeze=AVL_args.freeze_backbone)

    vl_transformer.to(AVL_args.device)
   
    if AVL_args.do_train:
        AVL_args = restore_training_settings(AVL_args)
        train_dataloader = make_data_loader(
            MAD_args,
            is_train=True,
            #is_distributed=distributed,
        )

        val_dataloader = None
        test_dataloader = None 
        test_period = MAD_args.SOLVER.TEST_PERIOD
        if test_period > 0:
            if len(MAD_args.DATASETS.VAL) != 0:
                val_dataloader = make_data_loader(MAD_args, is_train=False, is_for_period=True)
            else:
                logger.info('Please specify validation dataset in config file for performance evaluation during training')
            test_dataloader = make_data_loader(MAD_args, is_train=False)
        
        AVL_args.max_iter = len(train_dataloader)
        AVL_args.max_global_step = AVL_args.max_iter // AVL_args.gradient_accumulation_steps
        AVL_args.global_iters_per_epoch = AVL_args.max_global_step // AVL_args.num_train_epochs
        AVL_args.save_steps = AVL_args.global_iters_per_epoch * 3

        AVL_args, vl_transformer, optimizer, scheduler = mixed_precision_init(
            AVL_args, vl_transformer)
        train(MAD_args, AVL_args, train_dataloader, val_dataloader, vl_transformer,
              tokenizer, training_saver, optimizer, scheduler)


if __name__ == '__main__':

    # Import arguments for AVL transformer:
    shared_configs.shared_video_captioning_config(cbs=True, scst= True)
    AVL_args = get_custom_args(shared_configs)
    
    # Import arguments for MAD dataloader:
    MAD_config_file = 'src/configs/mad.yaml'
    MAD_args = get_cfg_defaults()
    MAD_args.merge_from_file(MAD_config_file)

    # Set up logging:
    logger = setup_logger(init_config, MAD_args)

    logger.info("Loaded configuration file {}".format(MAD_config_file))
    with open(MAD_config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    
    logger.info("MAD dataset running with config:\n{}".format(MAD_args))
    logger.info("AVL transformer running with config:\n{}".format(AVL_args))
    
    main(MAD_args, AVL_args)