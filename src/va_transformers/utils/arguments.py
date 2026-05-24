import argparse


# helper for reading in None from cli

def none_or_str(value):
    if value == 'None':
        return None
    return value


# arguments parser

class Arguments:
    def __init__(self, mode):
        self.parser = argparse.ArgumentParser(description='value-aware-transformers')
        self.mode = mode
        self.arguments = None

    def initialise(self):
        self.parser.add_argument('--config', type=str, required=True)

        # transformer specifications

        self.parser.add_argument('--token_emb_dim', type=int, default=100)
        self.parser.add_argument('--quant_emb_dim', type=int, default=5)
        self.parser.add_argument('--attn_dim', type=int, default=100)
        self.parser.add_argument('--attn_depth', type=int, default=4)
        self.parser.add_argument('--attn_heads', type=int, default=8)
        self.parser.add_argument('--attn_dim_quants', type=int, default=5)
        self.parser.add_argument('--attn_dropout', type=float, default=0.05)
        self.parser.add_argument('--ff_dropout', type=float, default=0.05)
        self.parser.add_argument('--use_rezero', type=int, default=0)
        self.parser.add_argument('--rotary_pos_emb', type=int, default=0)

        # general arguments

        self.parser.add_argument('--device', type=str)
        self.parser.add_argument('--toy_run', type=int, default=0)
        self.parser.add_argument('--WARNING_TESTING', type=int, default=0, choices=[0, 1])
        self.parser.add_argument('--model_name', type=str, default=f'{self.mode}_test')
        self.parser.add_argument('--num_epochs', type=int, default=2)
        self.parser.add_argument('--early_stopping_threshold', type=int, default=7)
        self.parser.add_argument('--batch_size_tr', type=int, default=100)
        self.parser.add_argument('--batch_size_val', type=int, default=100)
        self.parser.add_argument('--grad_accum_every', type=int, default=1)
        self.parser.add_argument('--seq_len', type=int, default=250)
        self.parser.add_argument('--write_best_val_embeddings', type=int, default=0)
        self.parser.add_argument('--write_initial_embeddings', type=int, default=0)
        self.parser.add_argument('--write_final_embeddings', type=int, default=0)
        self.parser.add_argument('--learning_rate', type=float, default=5e-5)
        self.parser.add_argument('--scheduler_decay', type=float, default=1.)
        self.parser.add_argument('--pad_token', type=int, default=0)
        self.parser.add_argument('--pad_quant_token', type=int, default=0)
        self.parser.add_argument('--specials', type=none_or_str, default='EOS')
        self.parser.add_argument('--align_sample_at', type=str, default='random/SOS',
                                 choices=['SOS', 'EOS', 'random/SOS', 'random/EOS'])
        self.parser.add_argument('--with_values', type=int, default=1)
        self.parser.add_argument('--va_transformer', type=int, default=1)
        self.parser.add_argument('--logit_head', type=none_or_str, default="hierarchical",
                                 choices=[None, 'shared', 'weak', 'separate', 'hierarchical'])

        self.parser.add_argument('--writer_flush_secs', type=int, default=120)

        # pretraining arguments

        if self.mode == 'pretraining':
            self.parser.add_argument('--mode', type=str, default='pretraining', choices=['pretraining', 'evaluation'])
            self.parser.add_argument('--gamma', type=float, default=0.9)
            self.parser.add_argument('--ignore_index', type=int, default=0)
            self.parser.add_argument('--ignore_quant_index', type=int, default=0)
            self.parser.add_argument('--load_from_checkpoint_at', type=str, default=None)

        # finetuning arguments

        if self.mode == 'finetuning':
            self.parser.add_argument('--mode', type=str, default='finetuning', choices=['finetuning', 'evaluation'])
            self.parser.add_argument('--pretrained_model', type=str, required=True)
            self.parser.add_argument('--load_from', type=str, default='pretrained')
            self.parser.add_argument('--targets', type=str, default="DEATH<=3D")
            self.parser.add_argument('--weighted_loss', type=int, default=1)
            self.parser.add_argument('--freeze_base', type=int, default=0)
            self.parser.add_argument('--clf_style', type=str, default='on_EOS',
                                     choices=['flatten', 'on_sample_start', 'on_sample_end', 'on_EOS', 'on_EOS-2'])
            self.parser.add_argument('--clf_hidden_dim', type=int, default=100)
            self.parser.add_argument('--clf_dropout', type=float, default=0.5)
            self.parser.add_argument('--clf_or_reg', type=str, default='clf', choices=['reg', 'clf'])
            self.parser.add_argument('--predict_on_train', type=int, default=0)
            self.parser.add_argument('--num_classes', type=int, default=2)
            self.parser.add_argument('--clf_depth', type=int, default=2)

        # baselining arguments

        elif self.mode == 'baselining':
            self.parser.add_argument('--targets', type=str, default="DEATH<=3D")
            self.parser.add_argument('--values_as', type=str, default='one-hot', choices=['int', 'one-hot'])
            self.parser.add_argument('--weighted_loss', type=int, default=1)
            self.parser.add_argument('--num_classes', type=int, default=2)
            self.parser.add_argument('--clf_dropout', type=float, default=0.5)
            self.parser.add_argument('--clf_hidden_dim', type=int, default=100)
            self.parser.add_argument('--clf_or_reg', type=str, default='clf', choices=['reg', 'clf'])
            self.parser.add_argument('--clf_depth', type=int, default=2)
            self.parser.add_argument('--mode', type=str, default='training', choices=['training'])
            self.parser.add_argument('--collapse_type', type=str, default='values_mean',
                                     choices=['values_mean', 'quants_mean'])

    def parse(self):
        self.initialise()
        self.arguments = self.parser.parse_args()
        return self.arguments
