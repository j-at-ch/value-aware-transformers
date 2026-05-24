class Mappings:
    def __init__(self,
                 mappings,
                 pad_token=None,
                 sos_token=None,
                 eos_token=None,
                 pad_quant_token=None,
                 sos_quant_token=None,
                 eos_quant_token=None
                 ):
        self.itemid2token = mappings['itemid2token']
        self.token2itemid = mappings['token2itemid']
        self.token2trcount = mappings['token2trcount']
        self.qname2qtoken = mappings['qname2qtoken']
        self.qtoken2qname = mappings['qtoken2qname']

        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.pad_quant_token = pad_quant_token
        self.sos_quant_token = sos_quant_token
        self.eos_quant_token = eos_quant_token

        if pad_token is not None:
            self.append_special_(self.itemid2token, self.token2itemid, '[PAD]', pad_token)
        if sos_token is not None:
            self.append_special_(self.itemid2token, self.token2itemid, '[SOS]', sos_token)
        if eos_token is not None:
            self.append_special_(self.itemid2token, self.token2itemid, '[EOS]', eos_token)

        if pad_quant_token is not None:
            self.append_special_(self.qname2qtoken, self.qtoken2qname, '[PAD]', pad_quant_token)
        if sos_quant_token is not None:
            self.append_special_(self.qname2qtoken, self.qtoken2qname, '[SOS]', sos_quant_token)
        if eos_quant_token is not None:
            self.append_special_(self.qname2qtoken, self.qtoken2qname, '[EOS]', eos_quant_token)

        self.num_tokens = len(self.itemid2token)
        self.num_quant_tokens = len(self.qname2qtoken) if self.qname2qtoken is not None else 0

    @staticmethod
    def append_special_(n2t, t2n, name, token):
        n2t[name] = token
        t2n[token] = name

    def top_n_train_tokens(self, n):
        d = sorted(self.token2trcount.items(), key=lambda item: item[1], reverse=True)
        return dict(d[0:n])

    def decode_token(self, token):
        return str(self.token2itemid[token])

    def decode_tokens(self, tokens):
        return ' '.join(list(map(self.decode_token, tokens)))


class Labellers:
    def __init__(self, mappings, d_items_df):
        self.mappings = mappings
        self.d_items_df = d_items_df

    def token2label(self, token):
        if token == self.mappings.pad_token:
            return '[PAD]'
        elif token == self.mappings.sos_token:
            return '[SOS]'
        elif token == self.mappings.eos_token:
            return '[EOS]'
        itemid = str(self.mappings.token2itemid[token])
        return self.d_items_df.loc[itemid, 'LABEL']

    def token2metadata(self, token):
        if token in (self.mappings.pad_token, self.mappings.sos_token, self.mappings.eos_token):
            name = self.mappings.token2itemid.get(token, str(token))
            return (str(name), '', '', 0)
        itemid = str(self.mappings.token2itemid[token])
        row = self.d_items_df.loc[itemid]
        count = self.mappings.token2trcount.get(token, 0)
        return (f"{row['LABEL']} ({row['FLUID']})", row['CATEGORY'], row['FLUID'], count)

    def tokens2label_string(self, tokens):
        return '\n\t -> '.join(list(map(self.token2label, tokens)))