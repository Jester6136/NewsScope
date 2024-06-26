from transformers.models.roberta.modeling_roberta import *
from dataclasses import dataclass
_CHECKPOINT_FOR_DOC = "roberta-base"
_CONFIG_FOR_DOC = "RobertaConfig"
_TOKENIZER_FOR_DOC = "RobertaTokenizer"

@dataclass
class EventExtractingModelOutput(QuestionAnsweringModelOutput):
    """
    Subclass extending QuestionAnsweringModelOutput to include event_logits.

    Args:
        event_logits (`torch.FloatTensor` of shape `(batch_size, num_events)`):
            Event logits.
    """
    event_logits: torch.FloatTensor = None

class MRCEventExtract(RobertaPreTrainedModel):
    config_class = RobertaConfig

    def _reorder_cache(self, past, beam_idx):
        pass

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, num_event_labels = 8, drop_prob = None):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_event_labels = num_event_labels

        self.roberta = RobertaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 8 basic classes

        # ["Business","Conflict","Contact","Justice","Life","Movement","Personnel","Transaction"]
        self.event_type_output = nn.Linear(config.hidden_size, self.num_event_labels)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            words_lengths=None,
            start_idx=None,
            end_idx=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            start_positions=None,
            end_positions=None,
            span_answer_ids=None,
            event_type_labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        features = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = features["last_hidden_state"]
        pooler_output = features["pooler_output"]
        context_embedding = sequence_output
        pooler_output = self.dropout(pooler_output)
        event_logits = self.event_type_output(pooler_output)

        # Compute align word sub_word matrix
        batch_size = input_ids.shape[0]
        max_sub_word = input_ids.shape[1]
        max_word = words_lengths.shape[1]
        align_matrix = torch.zeros((batch_size, max_word, max_sub_word))

        for i, sample_length in enumerate(words_lengths):
            for j in range(len(sample_length)):
                start_idx = torch.sum(sample_length[:j])
                align_matrix[i][j][start_idx: start_idx + sample_length[j]] = 1 if sample_length[j] > 0 else 0

        align_matrix = align_matrix.to(context_embedding.device)
        # Combine sub_word features to make word feature
        context_embedding_align = torch.bmm(align_matrix, context_embedding)

        logits = self.qa_outputs(context_embedding_align)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            
            if event_type_labels is not None:
                event_classify_loss = loss_fct(event_logits.view(-1, self.num_event_labels), event_type_labels.view(-1))
                total_loss = (start_loss + end_loss) / 2 + event_classify_loss
            else:
                total_loss = (start_loss + end_loss) / 2
            print(f"total_loss: {total_loss} start_loss: {start_loss}, end_loss: {end_loss}, event_classify_loss: {event_classify_loss}")


        if not return_dict:
            output = (start_logits, end_logits, event_logits) + sequence_output[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return EventExtractingModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            event_logits=event_logits,
            hidden_states=features.hidden_states,
            attentions=features.attentions,
        )


if __name__ == "__main__":
    # Example usage
    model_path = r"D:\NewsScope\model"
    model = MRCEventExtract.from_pretrained(model_path, num_event_labels = 8)
    print(model)