# -*- coding: utf-8 -*-
# 💾⚙️🔮

__author__ = "Daulet N."
__email__ = "daulet.nurmanbetov@gmail.com"

import logging
import torch
from simpletransformers.ner import NERModel


class RestorePuncts:
    def __init__(self, model_path: str, words_per_pred=250):
        if not model_path:
            raise ("You need to enter a model_path")
        self.words_per_pred = words_per_pred
        self.overlap_words = 30
        self.valid_labels = [
            "OU",
            "OO",
            ".O",
            "!O",
            ",O",
            ".U",
            "!U",
            ",U",
            ":O",
            ";O",
            ":U",
            "'O",
            "-O",
            "?O",
            "?U",
        ]
        self.model = NERModel(
            "bert",
            model_path,
            labels=self.valid_labels,
            use_cuda=torch.cuda.is_available(),
            args={"silent": True, "max_seq_length": 512},
        )

    def punctuate(self, text: str):
        """
        Performs punctuation restoration on arbitrarily large text.

        Args:
            - text (str): Text to punctuate, can be few words to as large as you want.
        """
        # split up large text into bert digestible chunks
        splits = self.split_on_tokens(text, self.words_per_pred, self.overlap_words)
        # predict slices
        # full_preds_list contains tuple of labels and logits
        full_preds_list = [self.predict(i["text"]) for i in splits]
        # extract predictions, and discard logits
        preds_list = [i[0][0] for i in full_preds_list]
        # join text slices
        combined_preds = self.combine_results(text, preds_list)
        # create punctuated prediction
        punct_text = self.punctuate_texts(combined_preds)
        return punct_text

    def predict(self, input_slice):
        """
        Passes the unpunctuated text to the model for punctuation.
        """
        predictions, raw_outputs = self.model.predict([input_slice])
        return predictions, raw_outputs

    @staticmethod
    def split_on_tokens(text, length, overlap):
        """
        Splits text into predefined slices of overlapping text with indexes (offsets)
        that tie-back to original text.
        This is done to bypass 512 token limit on transformer models by sequentially
        feeding chunks of < 512 tokens.
        Example output:
        [{...}, {"text": "...", 'start_idx': 31354, 'end_idx': 32648}, {...}]
        """
        words = text.replace("\n", " ").split(" ")
        resp = []
        list_chunk_idx = 0
        i = 0

        while True:
            # words in the chunk and the overlapping portion
            words_len = words[(length * i) : (length * (i + 1))]
            words_ovlp = words[(length * (i + 1)) : ((length * (i + 1)) + overlap)]
            words_split = words_len + words_ovlp

            # Break loop if no more words
            if not words_split:
                break

            words_str = " ".join(words_split)
            nxt_chunk_start_idx = len(" ".join(words_len))
            list_char_idx = len(" ".join(words_split))

            resp_obj = {
                "text": words_str,
                "start_idx": list_chunk_idx,
                "end_idx": list_char_idx + list_chunk_idx,
            }

            resp.append(resp_obj)
            list_chunk_idx += nxt_chunk_start_idx + 1
            i += 1
        logging.info(f"Sliced transcript into {len(resp)} slices.")
        return resp

    @staticmethod
    def combine_results(full_text: str, text_slices):
        """
        Given a full text and predictions of each slice combines predictions into a single text again.
        Performs validataion whether text was combined correctly
        """
        split_full_text = full_text.replace("\n", " ").split(" ")
        split_full_text = [i for i in split_full_text if i]
        split_full_text_len = len(split_full_text)
        output_text = []
        index = 0

        if len(text_slices[-1]) <= 3 and len(text_slices) > 1:
            text_slices = text_slices[:-1]

        for _slice in text_slices:
            slice_words = len(_slice)
            for ix, wrd in enumerate(_slice):
                if index == split_full_text_len:
                    break

                if (
                    split_full_text[index] == str(list(wrd.keys())[0])
                    and ix <= slice_words - 3
                    and text_slices[-1] != _slice
                ):
                    index += 1
                    pred_item_tuple = list(wrd.items())[0]
                    output_text.append(pred_item_tuple)
                elif (
                    split_full_text[index] == str(list(wrd.keys())[0])
                    and text_slices[-1] == _slice
                ):
                    index += 1
                    pred_item_tuple = list(wrd.items())[0]
                    output_text.append(pred_item_tuple)
        assert [i[0] for i in output_text] == split_full_text
        return output_text

    @staticmethod
    def punctuate_texts(full_pred: list):
        """
        Given a list of Predictions from the model, applies the predictions to text,
        thus punctuating it.
        """
        punct_resp = ""
        for i in full_pred:
            word, label = i
            if label[-1] == "U":
                punct_word = word.capitalize()
            else:
                punct_word = word

            if label[0] != "O":
                punct_word += label[0]

            punct_resp += punct_word + " "
        punct_resp = punct_resp.strip()
        # Append trailing period if doesn't exist.
        if punct_resp[-1].isalnum():
            punct_resp += "."
        return punct_resp
