# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import numpy as np


class Seq2Seq(nn.Module):
    """
        Build Seqence-to-Sequence.

        Parameters:

        * `encoder`- encoder of seq2seq model. e.g. roberta
        * `decoder`- decoder of seq2seq model. e.g. transformer
        * `config`- configuration of encoder model.
        * `beam_size`- beam size for beam search.
        * `max_length`- max length of target for beam search.
        * `sos_id`- start of symbol ids in target for beam search.
        * `eos_id`- end of symbol ids in target for beam search.
    """

    def __init__(self, encoder, decoder, config, beam_size=None, max_length=None, sos_id=None, eos_id=None,
                 device=None):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.register_buffer("bias", torch.tril(torch.ones(2048, 2048)))
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lsm = nn.LogSoftmax(dim=-1)
        self.tie_weights()

        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.criterion = nn.CrossEntropyLoss()
        self.device = device

    def avg_representation(self, hidden_states, attention_mask):
        """用于SimCSE训练的avg_representation"""
        first_hidden = hidden_states[1]
        last_hidden = hidden_states[-1]
        # print("first_hidden", first_hidden.shape)
        # print("attention_mask", attention_mask.shape)
        y = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(
            -1).unsqueeze(-1)

        # avg_last
        # y = (hidden_states * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)

        # CLS
        # y = hidden_states[:, 0]

        y = nn.functional.normalize(y, p=2, dim=1).to(self.device)
        return y

    def contrastive_loss(self, y_output, attn_mask, y_gold, t_attn_mask, temperature=0.05):
        """用于SimCSE训练的loss"""
        y_output = self.avg_representation(y_output, attn_mask).to(self.device)
        # print("y_output", y_output.shape)
        y_gold = self.avg_representation(y_gold, t_attn_mask).to(self.device)
        # print("y_gold", y_gold.shape)

        # whitening
        # kernel, bias = self.compute_kernel_bias([y_output,y_gold])
        # y = self.transform_and_normalize([y_output,y_gold], kernel, bias)
        # y_output = torch.tensor(y[0]).to(self.device)
        # y_gold = torch.tensor(y[1]).to(self.device)

        # 构造标签
        y_true = torch.arange(0, y_output.shape[0]).to(self.device)
        # 计算相似度
        # print("y_true", y_true.shape)
        similarities = torch.matmul(y_output, y_gold.permute([1, 0]).contiguous()).to(self.device)
        # print("similarities:", similarities.shape)
        similarities = similarities / temperature
        loss = self.criterion(similarities, y_true).to(self.device)
        return loss

    def _tie_or_clone_weights(self, first_module, second_module):
        """ Tie or clone module weights depending of weither we are using TorchScript or not
        """
        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head,
                                   self.encoder.embeddings.word_embeddings)

    def forward(self, tempers, source_ids, source_mask, position_idx, attn_mask, target_ids=None, target_mask=None,
                args=None):
        # embedding
        nodes_mask = position_idx.eq(0)
        # 用于比较两个 tensor 数据中是否相等 ( tensor类型), 并且可用于计算正确的个数
        token_mask = position_idx.ge(2)
        # torch.ge：实现大于等于（≥ \ge≥）运算
        inputs_embeddings = self.encoder.embeddings.word_embeddings(source_ids)
        # inputs_embeddings的维度是[batch_size, seq_len, hidden_size]
        nodes_to_token_mask = nodes_mask[:, :, None] & token_mask[:, None, :] & attn_mask
        nodes_to_token_mask = nodes_to_token_mask / (nodes_to_token_mask.sum(-1) + 1e-10)[:, :, None]
        avg_embeddings = torch.einsum("abc,acd->abd", nodes_to_token_mask, inputs_embeddings)
        # 矩阵乘法可以使用einsum计算为torch.einsum(“ij,jk->ik”, A, B)。这里，j 是求和下标，i 和 k 是输出下标
        inputs_embeddings = inputs_embeddings * (~nodes_mask)[:, :, None] + avg_embeddings * nodes_mask[:, :, None]

        outputs = self.encoder(inputs_embeds=inputs_embeddings, attention_mask=attn_mask, position_ids=position_idx,
                               output_hidden_states=True)
        # outputs[0]是最后一层的输出，outputs[1]是最后一层的隐藏状态,outputs[2]是所有层的隐藏状态
        encoder_output = outputs[0].permute([1, 0, 2]).contiguous()
        #  Tensor.permute(a,b,c,d, ...):permute函数可以对任意高维矩阵进行转置，参数是一个tuple，表示需要转置的维度
        # source_mask=token_mask.float()
        if target_ids is not None:
            tgt_embeddings = self.encoder.embeddings(target_ids)
            tgt_embeddings = tgt_embeddings.permute([1, 0, 2]).contiguous()

            tgt_outputs = self.encoder(target_ids, attention_mask=target_ids.ne(1), output_hidden_states=True)
            # tgt_embeddings的维度是[batch_size, seq_len, hidden_size]
            # tgt_embeddings经过转置之后的维度是[seq_len, batch_size, hidden_size]
            contra_loss = self.contrastive_loss(outputs[2], source_ids.ne(1),
                                                tgt_outputs[2], target_ids.ne(1), temperature=tempers)

            attn_mask = -1e4 * (1 - self.bias[:target_ids.shape[1], :target_ids.shape[1]])
            out = self.decoder(tgt_embeddings, encoder_output, tgt_mask=attn_mask,
                               memory_key_padding_mask=(1 - source_mask).bool())
            # out的维度是[seq_len, batch_size, hidden_size]
            hidden_states = torch.tanh(self.dense(out)).permute([1, 0, 2]).contiguous()
            # hidden_states的维度是[batch_size, seq_len, hidden_size]
            lm_logits = self.lm_head(hidden_states)
            # self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            # Shift so that tokens < n predict n
            active_loss = target_mask[..., 1:].ne(0).view(-1) == 1
            # PyTorch中的.view()函数是一个用于改变张量形状的方法。而不改变张量的数据,使用-1展平张量
            # torch.ne：实现不等于（≠ \ne≠）运算
            # active_loss的输出是一个一维的tensor，里面的值是True或者False
            # 把第二维的最后一个向量去掉，因为最后一个向量是[SEP]，不需要预测
            shift_logits = lm_logits[..., :-1, :].contiguous()
            # 把logits的第一个token去掉，因为第一个token是[CLS]，不需要预测
            shift_labels = target_ids[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                            shift_labels.view(-1)[active_loss])

            outputs = loss + contra_loss, (loss + contra_loss) * active_loss.sum(), active_loss.sum()
            return outputs
        else:
            # Predict
            preds = []
            zero = torch.cuda.LongTensor(1).fill_(0)
            for i in range(source_ids.shape[0]):
                context = encoder_output[:, i:i + 1]
                context_mask = source_mask[i:i + 1, :]
                beam = Beam(self.beam_size, self.sos_id, self.eos_id)
                input_ids = beam.getCurrentState()
                context = context.repeat(1, self.beam_size, 1)
                context_mask = context_mask.repeat(self.beam_size, 1)
                for _ in range(self.max_length):
                    if beam.done():
                        break
                    attn_mask = -1e4 * (1 - self.bias[:input_ids.shape[1], :input_ids.shape[1]])
                    tgt_embeddings = self.encoder.embeddings(input_ids).permute([1, 0, 2]).contiguous()
                    out = self.decoder(tgt_embeddings, context, tgt_mask=attn_mask,
                                       memory_key_padding_mask=(1 - context_mask).bool())
                    out = torch.tanh(self.dense(out))
                    hidden_states = out.permute([1, 0, 2]).contiguous()[:, -1, :]
                    out = self.lsm(self.lm_head(hidden_states)).data
                    # nn.LogSoftmax
                    beam.advance(out)
                    input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                    input_ids = torch.cat((input_ids, beam.getCurrentState()), -1)
                hyp = beam.getHyp(beam.getFinal())
                pred = beam.buildTargetTokens(hyp)[:self.beam_size]
                pred = [torch.cat([x.view(-1) for x in p] + [zero] * (self.max_length - len(p))).view(1, -1) for p in
                        pred]
                preds.append(torch.cat(pred, 0).unsqueeze(0))
                # unsqueeze(x)是在第x维添加维度的意思，
            preds = torch.cat(preds, 0)
            return preds


class Beam(object):
    def __init__(self, size, sos, eos):
        self.size = size
        self.tt = torch.cuda
        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size)
                       .fill_(0)]
        self.nextYs[0][0] = sos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))

        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >= self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished = []
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i))
            unfinished.sort(key=lambda a: -a[0])
            self.finished += unfinished[:self.size - len(self.finished)]
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps = []
        for _, timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j + 1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps

    def buildTargetTokens(self, preds):
        sentence = []
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok == self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence
