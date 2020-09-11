import torch
from torch import nn
from torch.nn import BCELoss, BCEWithLogitsLoss, CrossEntropyLoss, NLLLoss, MSELoss
from torch.nn import Dropout
from transformers import BertConfig, BertModel, BertForMaskedLM
from typing import Any

import torch.nn.functional as F

from collections import defaultdict

class BertPretrain(torch.nn.Module):
    def __init__(self,
                 model_name_or_path: str):
        super(BertPretrain, self).__init__()
        self.bert_model = BertForMaskedLM.from_pretrained(model_name_or_path)

    def forward(self, 
                input_ids: torch.tensor,
                mlm_labels: torch.tensor):
        outputs = self.bert_model(input_ids, masked_lm_labels=mlm_labels)
        return outputs[0]

class IntentBertModel(torch.nn.Module):
    def __init__(self,
                 model_name_or_path: str,
                 dropout: float,
                 num_intent_labels: int,
                 use_observers: bool = False):
        super(IntentBertModel, self).__init__()
        self.bert_model = BertModel.from_pretrained(model_name_or_path)

        self.dropout = Dropout(dropout)
        self.num_intent_labels = num_intent_labels
        self.intent_classifier = nn.Linear(self.bert_model.config.hidden_size, num_intent_labels)
        self.use_observers = use_observers

    def forward(self,
                input_ids: torch.tensor,
                # TODO (mihail): Should probably use a different attention mask since now computing loss with respect to first [PAD] token
                attention_mask: torch.tensor,
                token_type_ids: torch.tensor,
                intent_label: torch.tensor = None):
        if not self.use_observers:
            pooled_output = self.bert_model(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            token_type_ids=token_type_ids)[1]
        else:
            hidden_states = self.bert_model(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            token_type_ids=token_type_ids)[0]

            pooled_output =  hidden_states[:, -20:].mean(dim=1)


        intent_logits = self.intent_classifier(self.dropout(pooled_output))

        # Compute losses if labels provided
        if intent_label is not None:
            loss_fct = CrossEntropyLoss()
            intent_loss = loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_label.type(torch.long))
        else:
            intent_loss = torch.tensor(0)

        return intent_logits, intent_loss

class ExampleIntentBertModel(torch.nn.Module):
    def __init__(self,
                 model_name_or_path: str,
                 dropout: float,
                 num_intent_labels: int,
                 use_observers: bool = False):
        super(ExampleIntentBertModel, self).__init__()
        #self.bert_model = BertModel.from_pretrained(model_name_or_path)
        self.bert_model = BertModel(BertConfig.from_pretrained(model_name_or_path, output_attentions=True))

        self.dropout = Dropout(dropout)
        self.num_intent_labels = num_intent_labels
        self.use_observers = use_observers
        self.all_outputs = []

    def encode(self,
                input_ids: torch.tensor,
                attention_mask: torch.tensor,
                token_type_ids: torch.tensor):
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).repeat(1,1,input_ids.size(1),1)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.bert_model.parameters()).dtype) 

        # Combine attention maps
        padding = (input_ids.unsqueeze(1) == 0).unsqueeze(-1)
        padding = padding.repeat(1,1,1,padding.size(-2))

        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.bert_model.embeddings(input_ids, position_ids=None, token_type_ids=token_type_ids)
        encoder_outputs = self.bert_model.encoder(embedding_output,
                                                  extended_attention_mask,
                                                  head_mask=[None] * self.bert_model.config.num_hidden_layers)

        if encoder_outputs[0].size(0) == 1:
            pass
            #self.all_outputs.append(torch.cat(encoder_outputs[1], dim=0).cpu())
            #self.all_outputs.append(encoder_outputs[0][:, -20:].cpu())
        sequence_output = encoder_outputs[0]

        if self.use_observers:
            pooled_output = sequence_output[:, -20:].mean(dim=1)
        else:
            pooled_output = self.bert_model.pooler(sequence_output)

        return pooled_output


    def forward(self,
                input_ids: torch.tensor,
                attention_mask: torch.tensor,
                token_type_ids: torch.tensor,
                intent_label: torch.tensor,
                example_input: torch.tensor,
                example_mask: torch.tensor,
                example_token_types: torch.tensor,
                example_intents: torch.tensor):
        example_pooled_output = self.encode(input_ids=example_input,
                                            attention_mask=example_mask,
                                            token_type_ids=example_token_types)

        pooled_output = self.encode(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids)

        pooled_output = self.dropout(pooled_output)
        probs = torch.softmax(pooled_output.mm(example_pooled_output.t()), dim =-1)

        intent_probs = 1e-6 + torch.zeros(probs.size(0), self.num_intent_labels).cuda().scatter_add(-1, example_intents.unsqueeze(0).repeat(probs.size(0), 1), probs)

        # Compute losses if labels provided
        if intent_label is not None:
            loss_fct = NLLLoss()
            intent_lp = torch.log(intent_probs)
            intent_loss = loss_fct(intent_lp.view(-1, self.num_intent_labels), intent_label.type(torch.long))
        else:
            intent_loss = torch.tensor(0)

        return intent_probs, intent_loss

class SlotBertModel(torch.nn.Module):
    def __init__(self,
                 model_name_or_path: str,
                 dropout: float,
                 num_slot_labels: int,
                 use_observers: bool = False):
        super(SlotBertModel, self).__init__()
        self.bert_model = BertModel.from_pretrained(model_name_or_path)
        self.dropout = Dropout(dropout)
        self.num_slot_labels = num_slot_labels
        self.slot_classifier = nn.Linear(self.bert_model.config.hidden_size, num_slot_labels)
        self.use_observers = use_observers

    def encode(self,
               input_ids: torch.tensor,
               attention_mask: torch.tensor,
               token_type_ids: torch.tensor):
        if self.use_observers:
            input_ids = torch.cat([input_ids, input_ids], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.zeros(attention_mask.size()).cuda().long()], dim=-1)
            token_type_ids = torch.cat([token_type_ids, token_type_ids], dim=-1)
            hidden_states, _ = self.bert_model(input_ids=input_ids,
                                               attention_mask=attention_mask,
                                               token_type_ids=token_type_ids)
            return hidden_states[:, input_ids.size(1)//2:]
        else:
            hidden_states, _ = self.bert_model(input_ids=input_ids,
                                               attention_mask=attention_mask,
                                               token_type_ids=token_type_ids)
            return hidden_states

    def forward(self,
                input_ids: torch.tensor,
                attention_mask: torch.tensor,
                token_type_ids: torch.tensor,
                slot_labels: torch.tensor = None):
        hidden_states = self.encode(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids)

        slot_logits = self.slot_classifier(self.dropout(hidden_states))

        # Compute losses if labels provided
        if slot_labels is not None:
            loss_fct = CrossEntropyLoss()

            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                active_labels = slot_labels.view(-1)[active_loss]
                slot_loss = loss_fct(active_logits, active_labels.type(torch.long))
            else:
                slot_loss = loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels.view(-1).type(torch.long))
        else:
            slot_loss = torch.tensor(0).cuda() if torch.cuda.is_available() else torch.tensor(0)

        return slot_logits, slot_loss

class ExampleSlotBertModel(torch.nn.Module):
    def __init__(self,
                 model_name_or_path: str,
                 dropout: float,
                 num_slot_labels: int,
                 use_observers: bool = False):
        super(ExampleSlotBertModel, self).__init__()
        self.bert_model = BertModel.from_pretrained(model_name_or_path)
        self.dropout = Dropout(dropout)
        self.num_slot_labels = num_slot_labels
        self.slot_classifier = nn.Linear(self.bert_model.config.hidden_size, num_slot_labels)
        self.use_observers = use_observers

    def encode(self,
               input_ids: torch.tensor,
               attention_mask: torch.tensor,
               token_type_ids: torch.tensor):
        if self.use_observers:
            input_ids = torch.cat([input_ids, input_ids], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.zeros(attention_mask.size()).cuda().long()], dim=-1)
            token_type_ids = torch.cat([token_type_ids, token_type_ids], dim=-1)
            hidden_states, _ = self.bert_model(input_ids=input_ids,
                                               attention_mask=attention_mask,
                                               token_type_ids=token_type_ids)
            return hidden_states[:, input_ids.size(1)//2:]
        else:
            hidden_states, _ = self.bert_model(input_ids=input_ids,
                                               attention_mask=attention_mask,
                                               token_type_ids=token_type_ids)
            return hidden_states

    def forward(self,
                input_ids: torch.tensor,
                attention_mask: torch.tensor,
                token_type_ids: torch.tensor,
                slot_labels: torch.tensor,
                example_word_inds: torch.tensor,
                example_input: torch.tensor,
                example_mask: torch.tensor,
                example_token_types: torch.tensor,
                example_slots: torch.tensor):
        example_hidden_states = self.encode(input_ids=example_input,
                                            attention_mask=example_mask,
                                            token_type_ids=example_token_types)

        hidden_states = self.encode(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids)

        self.dropout(hidden_states)

        # relevant example states
        example_hidden = example_hidden_states[torch.arange(example_hidden_states.size(0)), example_word_inds]                                                                                                                         
        # Compute probabilities by copying from examples
        probs = torch.softmax(hidden_states.bmm( example_hidden.t().unsqueeze(0).repeat(hidden_states.size(0), 1, 1) ), dim=-1)                                                                                                                  
        example_slots = example_slots.view(1,1, example_slots.size(0)).repeat(probs.size(0), probs.size(1), 1)
        slot_probs = 1e-6 + torch.zeros(probs.size(0), probs.size(1), self.num_slot_labels).cuda().scatter_add(-1, example_slots, probs)

        # Compute losses if labels provided
        if slot_labels is not None:
            loss_fct = NLLLoss()

            slot_logits = torch.log(slot_probs)

            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                active_labels = slot_labels.view(-1)[active_loss]
                slot_loss = loss_fct(active_logits, active_labels.type(torch.long))
            else:
                slot_loss = loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels.view(-1).type(torch.long))
        else:
            slot_loss = torch.tensor(0).cuda() if torch.cuda.is_available() else torch.tensor(0)

        return slot_logits, slot_loss

class ResponseSelectionBertModel(torch.nn.Module):
    def __init__(self,
                 model_name_or_path: str,
                 dropout: float):
        super(ResponseSelectionBertModel, self).__init__()
        self.bert_model = BertModel.from_pretrained(model_name_or_path)

        self.dropout = Dropout(dropout)
        self.response_project = nn.Linear(self.bert_model.config.hidden_size, self.bert_model.config.hidden_size)

    def forward(self,
                ctx_input_ids: torch.tensor,
                ctx_attention_mask: torch.tensor,
                ctx_token_type_ids: torch.tensor,
                rsp_input_ids: torch.tensor,
                rsp_attention_mask: torch.tensor,
                rsp_token_type_ids: torch.tensor):
        _, ctx_pooled_output = self.bert_model(input_ids=ctx_input_ids,
                                           attention_mask=ctx_attention_mask,
                                           token_type_ids=ctx_token_type_ids)
        _, rsp_pooled_output = self.bert_model(input_ids=rsp_input_ids,
                                               attention_mask=rsp_attention_mask,
                                               token_type_ids=rsp_token_type_ids)

        rsp_pooled_output = self.response_project(self.dropout(rsp_pooled_output))

        rsp_logits = ctx_pooled_output.mm(rsp_pooled_output.t())
        rsp_labels = torch.arange(rsp_logits.size(0))

        loss_fct = CrossEntropyLoss()
        resp_loss = loss_fct(rsp_logits, rsp_labels.type(torch.long).cuda())

        return resp_loss

    def predict(self,
                ctx_input_ids: torch.tensor,
                ctx_attention_mask: torch.tensor,
                ctx_token_type_ids: torch.tensor,
                rsp_input_ids: torch.tensor,
                rsp_attention_mask: torch.tensor,
                rsp_token_type_ids: torch.tensor):
        _, ctx_pooled_output = self.bert_model(input_ids=ctx_input_ids,
                                           attention_mask=ctx_attention_mask,
                                           token_type_ids=ctx_token_type_ids)
        _, rsp_pooled_output = self.bert_model(input_ids=rsp_input_ids,
                                               attention_mask=rsp_attention_mask,
                                               token_type_ids=rsp_token_type_ids)

        rsp_pooled_output = self.response_project(self.dropout(rsp_pooled_output))

        rsp_logits = ctx_pooled_output.mm(rsp_pooled_output.t())
        return rsp_logits

class StateTrackingBertModel(torch.nn.Module):
    def __init__(self,
                 model_name_or_path: str,
                 dropout: float,
                 num_slot_labels: int):
        super(StateTrackingBertModel, self).__init__()
        self.bert_model = BertModel.from_pretrained(model_name_or_path)

        self.dropout = Dropout(dropout)
        self.num_slot_labels = num_slot_labels
        
        self.classifiers = nn.ModuleList([
            nn.Linear(self.bert_model.config.hidden_size, size) for size in num_slot_labels
        ])

    def forward(self,
                input_ids: torch.tensor,
                attention_mask: torch.tensor,
                token_type_ids: torch.tensor,
                state_label: torch.tensor = None):
        _, pooled_output = self.bert_model(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           token_type_ids=token_type_ids)


        loss_fct = CrossEntropyLoss()
        total_loss = 0
        logits = []
        if state_label is not None:
            for clf,label,num in zip(self.classifiers, state_label.t(), self.num_slot_labels):
                state_logits = clf(self.dropout(pooled_output))
                logits.append(state_logits)
                total_loss += loss_fct(state_logits.view(-1, num), label.type(torch.long))
        else:
            for clf in self.classifiers:
                state_logits = clf(self.dropout(pooled_output))
                logits.append(state_logits)

        return logits, total_loss

class JointSlotIntentBertModel(torch.nn.Module):
    def __init__(self,
                 model_name_or_path: str,
                 dropout: float,
                 num_intent_labels: int,
                 num_slot_labels: int):
        super(JointSlotIntentBertModel, self).__init__()
        self.bert_model = BertModel.from_pretrained(model_name_or_path)

        self.dropout = Dropout(dropout)
        self.num_intent_labels = num_intent_labels
        self.num_slot_labels = num_slot_labels
        self.intent_classifier = nn.Linear(self.bert_model.config.hidden_size, num_intent_labels)
        self.slot_classifier = nn.Linear(self.bert_model.config.hidden_size, num_slot_labels)

    def forward(self,
                input_ids: torch.tensor,
                # TODO (mihail): Should probably use a different attention mask since now computing loss with respect to first [PAD] token
                attention_mask: torch.tensor,
                token_type_ids: torch.tensor,
                intent_label: torch.tensor = None,
                slot_labels: torch.tensor = None):
        hidden_states, pooled_output = self.bert_model(input_ids=input_ids,
                                                       attention_mask=attention_mask,
                                                       token_type_ids=token_type_ids)


        intent_logits = self.intent_classifier(self.dropout(pooled_output))
        slot_logits = self.slot_classifier(self.dropout(hidden_states))

        # Compute losses if labels provided
        if slot_labels is not None:
            loss_fct = CrossEntropyLoss()

            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                active_labels = slot_labels.view(-1)[active_loss]
                slot_loss = loss_fct(active_logits, active_labels.type(torch.long))
            else:
                slot_loss = loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels.view(-1).type(torch.long))
        else:
            slot_loss = torch.tensor(0).cuda() if torch.cuda.is_available() else torch.tensor(0)

        # Compute losses if labels provided
        if intent_label is not None:
            loss_fct = CrossEntropyLoss()
            intent_loss = loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_label.type(torch.long))
        else:
            intent_loss = torch.tensor(0)

        return intent_logits, slot_logits, intent_loss + slot_loss

class ExampleJointSlotIntentBertModel(torch.nn.Module):
    def __init__(self,
                 model_name_or_path: str,
                 dropout: float,
                 num_intent_labels: int,
                 num_slot_labels: int):
        super(ExampleJointSlotIntentBertModel, self).__init__()
        self.bert_model = BertModel.from_pretrained(model_name_or_path)

        self.slot_model = ExampleSlotBertModel(model_name_or_path, dropout, num_slot_labels)
        self.intent_model = ExampleIntentBertModel(model_name_or_path, dropout, num_intent_labels)

        #self.slot_model.bert_model = self.bert_model
        #self.intent_model.bert_model = self.bert_model

    def encode(self,
                input_ids: torch.tensor,
                attention_mask: torch.tensor,
                token_type_ids: torch.tensor):
        pooled_output = self.intent_model.encode(input_ids, attention_mask, token_type_ids)
        hidden_states = self.slot_model.bert_model(input_ids=input_ids,
                                                   attention_mask=attention_mask,
                                                   token_type_ids=token_type_ids)[0]
        return hidden_states, pooled_output

    def forward(self,
                input_ids: torch.tensor,
                attention_mask: torch.tensor,
                token_type_ids: torch.tensor,
                intent_label: torch.tensor = None,
                slot_labels: torch.tensor = None,
                intent_examples: Any = None,
                slot_examples: Any = None):
        _, intent_loss = self.intent_model(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           token_type_ids=token_type_ids,
                                           intent_label=intent_label,
                                           example_input=intent_examples["input_ids"],
                                           example_mask=intent_examples["attention_mask"],
                                           example_token_types=intent_examples["token_type_ids"],
                                           example_intents=intent_examples["intent_label"])

        _, slot_loss = self.slot_model(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       token_type_ids=token_type_ids,
                                       slot_labels=slot_labels,
                                       example_word_inds=slot_examples["word_ind"],
                                       example_input=slot_examples["input_ids"],
                                       example_mask=slot_examples["attention_mask"],
                                       example_token_types=slot_examples["token_type_ids"],
                                       example_slots=slot_examples["slot_labels"])

        return intent_loss + slot_loss
