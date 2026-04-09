"""Wav2Vec2 model with integrated feature extractor for audio encoding."""

import torch.nn.functional as F
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Model as HF_Wav2Vec2Model
from transformers.modeling_outputs import BaseModelOutput

__all__ = ["Wav2Vec2Model"]


def linear_interpolation(features, seq_len):
    """Interpolate features to target sequence length."""
    features = features.transpose(1, 2)
    output_features = F.interpolate(features, size=seq_len, align_corners=True, mode="linear")
    return output_features.transpose(1, 2)


class Wav2Vec2Model(HF_Wav2Vec2Model):
    """Wav2Vec2 model with integrated audio preprocessor.

    Extends transformers Wav2Vec2Model with:
    - Integrated audio_processor (Wav2Vec2FeatureExtractor) for audio preprocessing
    - from_pretrained() that loads both model and audio processor automatically
    - Custom forward with linear interpolation for sequence length matching

    Note: self.feature_extractor is the internal CNN encoder (inherited from HF).
          self.audio_processor is the Wav2Vec2FeatureExtractor for audio preprocessing.
    """

    audio_processor: Wav2Vec2FeatureExtractor  # Audio preprocessor (normalization, padding, etc.)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """Load model with integrated audio processor from pretrained.

        Args:
            pretrained_model_name_or_path: Path to pretrained model
            **kwargs: Additional arguments passed to HF from_pretrained

        Returns:
            Wav2Vec2Model with audio_processor attribute set
        """
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

        # Load and attach audio preprocessor
        model.audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        return model

    def forward(
        self,
        input_values,
        seq_len,
        attention_mask=None,
        mask_time_indices=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        self.config.output_attentions = False

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)
        extract_features = linear_interpolation(extract_features, seq_len=seq_len)

        if attention_mask is not None:
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        hidden_states, extract_features = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def feature_extract(self, input_values, seq_len):
        """Extract features from raw audio."""
        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)
        extract_features = linear_interpolation(extract_features, seq_len=seq_len)
        return extract_features

    def encode(
        self,
        extract_features,
        attention_mask=None,
        mask_time_indices=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """Encode pre-extracted features through transformer."""
        self.config.output_attentions = True

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if attention_mask is not None:
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        hidden_states, extract_features = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
